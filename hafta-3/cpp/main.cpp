#include <hip/hip_runtime.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numbers>
#include <random>
#include <cstdio>
#include <fstream>
#include <sstream>

#define UNODE_DEVICE_BUFFER_ENABLE_OVERFLOW_WARNINGS true

#define UValueNodePool_chunk_allocate_count (2048ULL * 32ULL)
#define UValueNodePool_gpu_chunk_allocate_count (2048ULL * 128ULL) // larger for not wasting gpu

#define dbg() (std::cout << "hit@" << __FILE__ << ":" << __LINE__ << "\n");

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if(err != hipSuccess){ \
        std::cerr << "HIP ERROR at " << __FILE__ << ":" << __LINE__ << " -> " << hipGetErrorString(err) << "\n"; \
    } \
} while(0)

float random_float(float min, float max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(min, max);

    return dist(gen);
}

typedef enum{
    UNDEFINED_OPERATION,
    ADD,
    SUB,
    DIV,
    MUL,
    TANH,
    POW,
    LOG,
    SIGMOID,
}operation_type;

// forward decs
class uValue;
class uValueNode;

#define INVALID_NODE 0ULL

struct uNodePool{
    std::vector<uValueNode> uValue_pool;
    uint64_t next = 1; // last available node (node 0 reserved for overflow handling)
    uint64_t temp_start = 0;
    bool temp_enabled = false;
};
uNodePool pool;

void uNodePool_expand_pool(size_t expansion_count){
    pool.uValue_pool.resize(pool.uValue_pool.size() + expansion_count);
}

// temp pool is a basic garbage collector for training loops etc.
void uNodePool_temp_begin(){
    if(!pool.temp_enabled){
        pool.temp_start = pool.next;
        pool.temp_enabled = true;
    }else{
        std::cout << "WARNING! temp pool started before last pool ended!\n";
    }
}

void uNodePool_temp_end(){
    if(pool.temp_enabled){
        pool.next = pool.temp_start;
        pool.temp_enabled = false;
    }else{
        std::cout << "WARNING! temp pool ended before a pool started!\n";
    }
}

struct gpu_uNodePool{
    uValueNode* uValue_pool;
    uint64_t pool_size; // count of nodes
    uint64_t next; // last available node

    // this two variable is not used
    // they will be implemented for optimisation in the feature
    uint64_t overflow_index;
    uint64_t overflowed;
};
__device__ gpu_uNodePool gpu_pool;
gpu_uNodePool gpu_pool_host_copy;

typedef unsigned long long uNodeIdx; // uValueNode index in pool table

struct operation_data
{
        uNodeIdx left = INVALID_NODE;
        uNodeIdx right = INVALID_NODE;
        operation_type op_type = operation_type::UNDEFINED_OPERATION;
};

class uValueNode{
    public:
        operation_data parent_operation;
        float value = 0.0f;
        float grad = 0.0f;

        __host__
        uNodeIdx create_copy(){
            // get new node from pool
            uNodeIdx new_node_idx = pool.next++;
            // check pool is large enough
            if(new_node_idx >= pool.uValue_pool.size()){
                pool.uValue_pool.resize(pool.uValue_pool.size() + UValueNodePool_chunk_allocate_count);
            }
            pool.uValue_pool[new_node_idx] = *this;
            return new_node_idx;
        }

        __device__
        uNodeIdx create_copy(){
            // get new node from pool
            uNodeIdx new_node_idx = atomicAdd(&(gpu_pool.next), 1ULL);
            // check pool is large enough
            if(new_node_idx >= gpu_pool.pool_size){
                // report over flow to cpu
                atomicExch(&(gpu_pool.overflowed), true);
                atomicMin(&gpu_pool.overflow_index, blockIdx.x * blockDim.x + threadIdx.x);
                printf("pool over flowed\n");
                __builtin_trap();
                return INVALID_NODE;
            }
            gpu_pool.uValue_pool[new_node_idx] = *this;
            return new_node_idx;
        }

        __host__
        static uNodeIdx new_node(){
            // get new node from pool
            uNodeIdx new_node_idx = pool.next++;
            // check pool is large enough
            if(new_node_idx >= pool.uValue_pool.size()){
                pool.uValue_pool.resize(pool.uValue_pool.size() + UValueNodePool_chunk_allocate_count);
                printf("pool over flowed2\n");
            }
            return new_node_idx;
        }

        __device__
        static uNodeIdx new_node(){
            // get new node from pool
            uNodeIdx new_node_idx = atomicAdd(&(gpu_pool.next), 1ULL);
            // check pool is large enough
            if(new_node_idx >= gpu_pool.pool_size){
                // report over flow to cpu
                printf("pool over flowed3\n");
                atomicExch(&(gpu_pool.overflowed), true);
                atomicMin(&gpu_pool.overflow_index, blockIdx.x * blockDim.x + threadIdx.x);
                printf("pool over flowed4\n");
                __builtin_trap();
                return INVALID_NODE;
            }
            return new_node_idx;
        }
        
        __host__
        static uValueNode* get_node(uNodeIdx idx){
            return &(pool.uValue_pool[idx]);
        }

        __device__
        static uValueNode* get_node(uNodeIdx idx){
            if(idx >= gpu_pool.pool_size){
                printf("blowed up\n");
                __builtin_trap();
            }
            return &(gpu_pool.uValue_pool[idx]);
        }

};  

bool uNodePool_copy_device_to_host(){
    // firstly get device pool
    HIP_CHECK(hipMemcpyFromSymbol(&gpu_pool_host_copy, HIP_SYMBOL(gpu_pool), sizeof(gpu_uNodePool), 0, hipMemcpyDeviceToHost));

    if(gpu_pool_host_copy.overflowed){
        #if UNODE_DEVICE_BUFFER_ENABLE_OVERFLOW_WARNINGS
        std::cout << "WARNING! gpu buffer overflowed!\n";
        #endif

        uNodePool_expand_pool(UValueNodePool_gpu_chunk_allocate_count);
        gpu_pool_host_copy.overflowed = false;
        return false;
    }

    // then copy pool content in to host pool
    pool.uValue_pool.resize(gpu_pool_host_copy.pool_size);
    HIP_CHECK(hipMemcpy(pool.uValue_pool.data(), gpu_pool_host_copy.uValue_pool, sizeof(uValueNode) * pool.uValue_pool.size(), hipMemcpyDeviceToHost));

    pool.next = gpu_pool_host_copy.next;

    return true;

    // done!
}

void uNodePool_copy_host_to_device(){
    // firstly update pool info
    if(!pool.uValue_pool.empty()){
        if(gpu_pool_host_copy.pool_size != pool.uValue_pool.size()){
            gpu_pool_host_copy.pool_size = pool.uValue_pool.size();

            if(gpu_pool_host_copy.uValue_pool == nullptr){
                HIP_CHECK(hipMalloc(&gpu_pool_host_copy.uValue_pool, gpu_pool_host_copy.pool_size * sizeof(uValueNode)));
            }else{
                HIP_CHECK(hipFree(gpu_pool_host_copy.uValue_pool));
                gpu_pool_host_copy.uValue_pool = nullptr;
                HIP_CHECK(hipMalloc(&gpu_pool_host_copy.uValue_pool, gpu_pool_host_copy.pool_size * sizeof(uValueNode)));
            }

        }

        if(gpu_pool_host_copy.uValue_pool == nullptr){
            std::cout << "ERROR: Failed to allocate gpu node table!\n";
        }
    }else{
        HIP_CHECK(hipFree(gpu_pool_host_copy.uValue_pool));
        gpu_pool_host_copy.pool_size = 0;
        gpu_pool_host_copy.uValue_pool = nullptr;
    }

    gpu_pool_host_copy.next = pool.next;

    // then copy pool content in to device pool
    HIP_CHECK(hipMemcpy(gpu_pool_host_copy.uValue_pool, pool.uValue_pool.data(), sizeof(uValueNode) * gpu_pool_host_copy.pool_size, hipMemcpyHostToDevice));

    // finaly copy host pool to device
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(gpu_pool), &gpu_pool_host_copy, sizeof(gpu_uNodePool), 0, hipMemcpyHostToDevice));

    // done!
}

class uValue{
    public:

        uNodeIdx node = INVALID_NODE;

        __host__
        uValue(float val){
            node = uValueNode::new_node();
            uValueNode::get_node(this->node)->value = val;
            uValueNode::get_node(this->node)->grad = 0.0f;
        }

        __device__
        uValue(float val){
                        printf("n1\n");
                __builtin_trap();
            node = uValueNode::new_node();
            uValueNode::get_node(this->node)->value = val;
            uValueNode::get_node(this->node)->grad = 0.0f;
        }

        __host__ __device__
        uValue(){
            node = uValueNode::new_node();
            uValueNode::get_node(this->node)->value = 0.0f;
            uValueNode::get_node(this->node)->grad = 0.0f;
        }

        // _this_arg_is_for_tricking_compiler_ is for shuting up compiler about ambigous
        __host__
        uValue(uNodeIdx node_, int _this_arg_is_for_tricking_compiler_){
            if(node_ >= pool.uValue_pool.size()){
                printf("bruh1\n");
            }
            node = node_;
        }

        __device__
        uValue(uNodeIdx node_, int _this_arg_is_for_tricking_compiler_){
            if(node_ >= gpu_pool.pool_size){
                printf("bruh2\n");
            }
            node = node_;
        }

        // initalize with existing node
        __host__ __device__
        static uValue from_node(uNodeIdx node_){
            return uValue(node_, 0);
        }

        __host__
        static void flush_uValue_pool(){
            pool.uValue_pool.clear();
        }

        __host__
        static void reset_uValue_backward_pool(){
            for(size_t i = 0; i < pool.uValue_pool.size(); i++){
                pool.uValue_pool[i].grad = 0.0f;
                pool.uValue_pool[i].parent_operation = operation_data(); // clear operation
            }
        }

        __host__ __device__
        float get_value(){
            return uValueNode::get_node(this->node)->value;
        }

        __host__ __device__
        float get_grad(){
            return uValueNode::get_node(this->node)->grad;
        }

        __host__ __device__
        uValue tanh(){

            // tanh formula
            // NOTE: this formula is not  optimised  for float. IT OVER FLOWS!
            // maybe clamping output to betweeen -1 & 1 solves problem but who cares
            // uValue child((std::pow(M_E, this->node->value) - std::pow(M_E, -(this->node->value))) / (std::pow(M_E, this->node->value) + std::pow(M_E, -(this->node->value))));

            uValue child(::tanh(uValueNode::get_node(this->node)->value));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = INVALID_NODE;
            op.op_type = operation_type::TANH;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }
        
        __host__ __device__
        uValue sigmoid(){

            // sigmoid formula
            uValue child(1.0f / (1.0f + ::exp(-(uValueNode::get_node(this->node)->value))));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = INVALID_NODE;
            op.op_type = operation_type::SIGMOID;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue pow(uValue power){

            uValue child(::pow(uValueNode::get_node(this->node)->value, uValueNode::get_node(power.node)->value));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = power.node;
            op.op_type = operation_type::POW;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }
       
        __host__ __device__
        uValue log(){
            float v = uValueNode::get_node(this->node)->value;

            float safe_v = v < 1e-7f ? 1e-7f : v;

            uValue child(::log(safe_v));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = INVALID_NODE;
            op.op_type = operation_type::LOG;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue operator+(const uValue& other){
            uValue child(uValueNode::get_node(this->node)->value + uValueNode::get_node(other.node)->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::ADD;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue& operator+=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = uValueNode::get_node(this->node)->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::ADD;

            uValueNode::get_node(this->node)->value += uValueNode::get_node(other.node)->value;

            uValueNode::get_node(this->node)->parent_operation = operation_data(); // old history holded by copy so we should clear this history for preventing double referance
            uValueNode::get_node(this->node)->parent_operation = op;

            return *this;
        }

        __host__ __device__
        uValue operator-(const uValue& other){
            uValue child(uValueNode::get_node(this->node)->value - uValueNode::get_node(other.node)->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::SUB;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue& operator-=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = uValueNode::get_node(this->node)->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::SUB;

            uValueNode::get_node(this->node)->value -= uValueNode::get_node(other.node)->value;

            uValueNode::get_node(this->node)->parent_operation = operation_data(); // old history holded by copy so we should clear this history for preventing double referance
            uValueNode::get_node(this->node)->parent_operation = op;

            return *this;
        }

        __host__ __device__
        uValue operator*(const uValue& other){
            uValue child(uValueNode::get_node(this->node)->value * uValueNode::get_node(other.node)->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::MUL;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue& operator*=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = uValueNode::get_node(this->node)->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::MUL;

            uValueNode::get_node(this->node)->value *=uValueNode::get_node(other.node)->value;

            uValueNode::get_node(this->node)->parent_operation = operation_data(); // old history holded by copy so we should clear this history for preventing double referance
            uValueNode::get_node(this->node)->parent_operation = op;

            return *this;
        }

        __host__ __device__
        uValue operator/(const uValue& other){
            uValue child(uValueNode::get_node(this->node)->value / uValueNode::get_node(other.node)->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::DIV;

            uValueNode::get_node(child.node)->parent_operation = op;

            return child;
        }

        __host__ __device__
        uValue& operator/=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = uValueNode::get_node(this->node)->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::DIV;

            uValueNode::get_node(this->node)->value /= uValueNode::get_node(other.node)->value;

            uValueNode::get_node(this->node)->parent_operation = operation_data(); // old history holded by copy so we should clear this history for preventing double referance
            uValueNode::get_node(this->node)->parent_operation = op;

            return *this;
        }

        __host__ __device__
        uValue& operator=(const uValue& other){
            uValueNode::get_node(this->node)->value =  uValueNode::get_node(other.node)->value;
            uValueNode::get_node(this->node)->grad = uValueNode::get_node(other.node)->grad;
            uValueNode::get_node(this->node)->parent_operation = uValueNode::get_node(other.node)->parent_operation;
            return *this;
        }

        // float is out side of our scope
        __host__ __device__
        uValue& operator=(const float other){
            uValueNode::get_node(this->node)->value = other;
            uValueNode::get_node(this->node)->grad = 0.0f;
            uValueNode::get_node(this->node)->parent_operation = operation_data();
            return *this;
        }

        __host__
        std::string dump_parents(){
            std::string result = "";
            if(uValueNode::get_node(this->node)->parent_operation.op_type == operation_type::UNDEFINED_OPERATION){
                result += "{[val]: " + std::to_string(uValueNode::get_node(this->node)->value) + " [grad]: " + std::to_string(uValueNode::get_node(this->node)->grad) + "}";
            }


            switch (uValueNode::get_node(this->node)->parent_operation.op_type)
            {
            case ADD:
                result += "(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + " + " + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).dump_parents() + ")";
                break;  

            case SUB:
                result += "(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + " - " + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).dump_parents() + ")";
                break;

            case MUL:
                result += "(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + " * " + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).dump_parents() + ")";
                break;

            case DIV:
                result += "(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + " / " + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).dump_parents() + ")";
                break;

            case POW:
                result += ("pow(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + " ," + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).dump_parents() + ")");
                break;                    

            case TANH:
                result += ("tanh(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + ")");
                break; 

            case LOG:
                result += ("log(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + ")");
                break;

            case SIGMOID:
                result += ("sigmoid(" + uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).dump_parents() + ")");
                break;
            
            default:
                break;
            }
            

            return result;
        }

        // backward gradient calculation
        __host__
        void backward(bool root = true){

            if(root){
                uValueNode::get_node(this->node)->grad = 1.0f;
            }
            
            // calculate gradians
            switch (uValueNode::get_node(this->node)->parent_operation.op_type)
            {
            case ADD:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad += uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
                break;  

            case SUB:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad -= uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
                break;

            case MUL:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value * uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad += uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
                break;

            case DIV:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += (1.0f / uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value) * uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad += (-(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value) / std::pow(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value, 2)) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
                break;

            case POW:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value * ::pow(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value, uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value - 1.0f) * uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad += uValueNode::get_node(this->node)->value * ::log(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
                break;

            case TANH:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += (1.0f - std::pow(uValueNode::get_node(this->node)->value, 2)) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                break;

            case LOG:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += (1.0f / uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                break;  

            case SIGMOID:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += (uValueNode::get_node(this->node)->value * (1.0f - uValueNode::get_node(this->node)->value)) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                break;
            
            default:
                break;
            }
        
        }
    
};

class uNeuron{
    public:
        uValue* weights;
        uValue* weights_device_buffer;
        size_t weight_count;
        uValue bias = 1.0f;

        static uNeuron create_neuron(size_t input_count){
            uNeuron neuron;
            neuron.weight_count = input_count;
            neuron.weights = new uValue[neuron.weight_count];
            HIP_CHECK(hipMalloc(&(neuron.weights_device_buffer), neuron.weight_count * sizeof(uValue)));

            if(neuron.weights_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu neuron buffer!\n";
            }

            for(size_t i = 0; i < neuron.weight_count; i++){
                neuron.weights[i] = random_float(0.01f, 0.7f);
            }
            neuron.bias = random_float(0.1f, 1.0f);
            return neuron;
        }

        void copy_neuron_to_device(){
            HIP_CHECK(hipMemcpy(weights_device_buffer, weights, sizeof(uValue) * weight_count, hipMemcpyHostToDevice));
        }

        void copy_neuron_to_host(){
            HIP_CHECK(hipMemcpy(weights, weights_device_buffer, sizeof(uValue) * weight_count, hipMemcpyDeviceToHost));
        }
};

__global__
void _forward_layer_gpu_thread_(uValue* input, uValue output_start, uNeuron* neurons, size_t neuron_count){
                printf("n1\n");
                __builtin_trap();
     uValue y = 0.0f;
    /*
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < neuron_count){
        // calculate neuron output
        uValue y = 0.0f;
        for(size_t j = 0; j < neurons[id].weight_count; j++){
            y += (neurons[id].weights_device_buffer[j] * input[j]);
        }
        y += neurons[id].bias;
        uValue::from_node(output_start.node + id) = y.sigmoid();
    }
    */
}

__global__
void _train_neuron_gpu_thread_(float learning_rate, uNeuron* neurons, size_t neuron_count){
    /*
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < neuron_count){
        for(size_t j = 0; j < neurons[id].weight_count; j++){
            neurons[id].weights_device_buffer[j] = (neurons[id].weights_device_buffer[j] - (neurons[id].weights_device_buffer[j].get_grad() * learning_rate)).get_value();
        }
        neurons[id].bias = (neurons[id].bias - (neurons[id].bias.get_grad() * learning_rate)).get_value();
    }
    */
}

typedef enum{
    UNDEFINED_NNETWORK,
    MLP,
    RNN,
}uNNetwork_type;

class uLayer{
    public:
        uNNetwork_type type = uNNetwork_type::UNDEFINED_NNETWORK;
        size_t input_count = 0;
        size_t neuron_count = 0;
        uNeuron* neurons = nullptr;
        uNeuron* neurons_device_buffer = nullptr;

        // rnn specific
        size_t total_input_count = 0;
        size_t h_input_count = 0;
        uValue* rnn_total_input = nullptr;

        static uLayer create_mlp_layer(size_t input_count, size_t neuron_count){
            uLayer layer;
            layer.type = uNNetwork_type::MLP;
            layer.input_count = input_count;
            layer.neuron_count = neuron_count;
            layer.neurons = new uNeuron[layer.neuron_count];
            layer.total_input_count  = input_count;
            layer.h_input_count = 0;
            layer.rnn_total_input = nullptr;

            HIP_CHECK(hipMalloc(&(layer.neurons_device_buffer), neuron_count * sizeof(uNeuron)));

            if(layer.neurons_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu neurons buffer!\n";
            }

            for(size_t i = 0; i < neuron_count; i++){
                layer.neurons[i] = uNeuron::create_neuron(layer.total_input_count);
            }

            return layer;
        }

        static uLayer create_rnn_layer(size_t input_count, size_t neuron_count){
            uLayer layer;
            layer.type = uNNetwork_type::RNN;
            layer.input_count = input_count;
            layer.neuron_count = neuron_count;
            layer.neurons = new uNeuron[layer.neuron_count];
            layer.total_input_count  = input_count + neuron_count;
            layer.h_input_count = neuron_count;

            layer.rnn_total_input = new uValue[layer.total_input_count];

            HIP_CHECK(hipMalloc(&(layer.neurons_device_buffer), neuron_count * sizeof(uNeuron)));

            if(layer.neurons_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu neurons buffer!\n";
            }

            for(size_t i = 0; i < neuron_count; i++){
                layer.neurons[i] = uNeuron::create_neuron(layer.total_input_count);
            }

            return layer;
        }

        void copy_neurons_to_device(){
            for(size_t i = 0; i < neuron_count; i++){
                neurons[i].copy_neuron_to_device();
            }
            HIP_CHECK(hipMemcpy(neurons_device_buffer, neurons, neuron_count * sizeof(uNeuron), hipMemcpyHostToDevice));
        }

        void copy_neurons_to_host(){
            for(size_t i = 0; i < neuron_count; i++){
                neurons[i].copy_neuron_to_host();
            }
            HIP_CHECK(hipMemcpy(neurons, neurons_device_buffer, neuron_count * sizeof(uNeuron), hipMemcpyDeviceToHost));
        }

        __host__
        void train(float learning_rate){
            copy_neurons_to_device();
            
            while(true){
                uNodePool_copy_host_to_device();

                size_t threads_per_block = 256;
                size_t blocks = (neuron_count + threads_per_block - 1) / threads_per_block;

                hipLaunchKernelGGL(_train_neuron_gpu_thread_, dim3(blocks), dim3(threads_per_block), 0, 0, learning_rate, this->neurons_device_buffer, this->neuron_count);

                HIP_CHECK(hipDeviceSynchronize());

                if(uNodePool_copy_device_to_host()){
                    copy_neurons_to_host();
                    break;
                }
            }
        }

        void reset_hidden_rnn(){
            if(this->type == uNNetwork_type::RNN){
                for(size_t i = this->input_count; i < this->total_input_count; i++){
                    this->rnn_total_input[i] = 0.0f;
                }
            }
        }

        std::vector<uValue> forward_layer(uValue* input){
            std::vector<uValue> output(this->neuron_count); // output

            // copy input to total buffer if layer is RNN
            if(this->type == uNNetwork_type::RNN){
                for(size_t i = 0; i < this->input_count; i++){
                    this->rnn_total_input[i] = input[i];
                }
            }

            // input
            uValue* input_device_buffer = nullptr;
            HIP_CHECK(hipMalloc(&input_device_buffer, sizeof(uValue) * this->total_input_count));
            if(input_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu input device buffer!\n";
                return output;
            }
            // copy total buffer to device if layer is RNN
            if(this->type == uNNetwork_type::RNN){
                HIP_CHECK(hipMemcpy(input_device_buffer, this->rnn_total_input, sizeof(uValue) * this->total_input_count, hipMemcpyHostToDevice));
            }else{
                HIP_CHECK(hipMemcpy(input_device_buffer, input, sizeof(uValue) * this->total_input_count, hipMemcpyHostToDevice));
            }
            
            copy_neurons_to_device();

            size_t threads_per_block = 256;
            size_t blocks = (neuron_count + threads_per_block - 1) / threads_per_block;

            while(true){

                // now copy to device
                uNodePool_copy_host_to_device();

                hipLaunchKernelGGL(_forward_layer_gpu_thread_, dim3(blocks), dim3(threads_per_block), 0, 0, input_device_buffer, output[0], this->neurons_device_buffer, this->neuron_count);

                HIP_CHECK(hipDeviceSynchronize());

                // read back to host
                if(uNodePool_copy_device_to_host() == true){
                    copy_neurons_to_host();

                    // copy output to total buffer if layer is RNN
                    // note: get value used because we dont want to copy history
                    if(this->type == uNNetwork_type::RNN){
                        for(size_t i = this->input_count; i < this->total_input_count; i++){
                            this->rnn_total_input[i] = output[i - this->input_count].get_value();
                        }
                    }

                    break;
                }
                #if UNODE_DEVICE_BUFFER_ENABLE_OVERFLOW_WARNINGS
                std::cout << "WARN! Forward failed because of overflow!\n";
                #endif
            }

            HIP_CHECK(hipFree(input_device_buffer));
            return output;
        }
};

typedef enum{
    UNDEFINED_LOSS_TYPE,
    MSE,
    NLL,
}uNNetwork_loss_type;

class uNNetwork{
    public:
        uLayer input_layer;
        std::vector<uLayer> hidden_layers;
        uLayer output_layer;
        uNNetwork_type type = uNNetwork_type::UNDEFINED_NNETWORK;

        static uNNetwork create_MLP(size_t input_count, size_t input_neuron_count, size_t output_neuron_count, size_t* hidden_layer_neuron_counts, size_t hidden_layer_count){
            uNNetwork network;
            network.type = uNNetwork_type::MLP;
            network.input_layer = uLayer::create_mlp_layer(input_count, input_neuron_count);

            size_t last_layer_output_count = input_neuron_count;

            for(size_t i = 0; i < hidden_layer_count; i++){
                uLayer hidden = uLayer::create_mlp_layer(last_layer_output_count, hidden_layer_neuron_counts[i]);
                network.hidden_layers.push_back(hidden);
                last_layer_output_count = hidden_layer_neuron_counts[i];
            }

            network.output_layer = uLayer::create_mlp_layer(last_layer_output_count, output_neuron_count);

            return network;
        }

        static uNNetwork create_RNN(size_t input_count, size_t input_neuron_count, size_t output_neuron_count, size_t* hidden_layer_neuron_counts, size_t hidden_layer_count){
            uNNetwork network;
            network.type = uNNetwork_type::RNN;
            network.input_layer = uLayer::create_rnn_layer(input_count, input_neuron_count);

            size_t last_layer_output_count = input_neuron_count;

            for(size_t i = 0; i < hidden_layer_count; i++){
                uLayer hidden = uLayer::create_rnn_layer(last_layer_output_count, hidden_layer_neuron_counts[i]);
                network.hidden_layers.push_back(hidden);
                last_layer_output_count = hidden_layer_neuron_counts[i];
            }

            network.output_layer = uLayer::create_rnn_layer(last_layer_output_count, output_neuron_count);

            return network;
        }

        void reset_hidden_rnn(){
            if(this->type == uNNetwork_type::RNN){
                input_layer.reset_hidden_rnn();
                for(size_t i = 0; i < hidden_layers.size(); i++){
                    hidden_layers[i].reset_hidden_rnn();
                }
                output_layer.reset_hidden_rnn();
            }
        }

        // NOTE: check layer sizes otherwise this function could cause segfault
        std::vector<uValue> forward(float* input){
            // convert inputs to uValue
            std::vector<uValue> input_values;
            for(size_t i = 0; i < input_layer.input_count; i++){
                input_values.push_back(uValue(input[i]));
            }

            // input layer
            std::vector<uValue> out = input_layer.forward_layer(input_values.data());

            // hidden layers
            for(size_t i = 0; i < hidden_layers.size(); i++){
                out = hidden_layers[i].forward_layer(out.data());
            }

            // output layer
            out = output_layer.forward_layer(out.data());

            return out;
        }

        void backward_training(float* target, uValue* output, float learning_rate, uNNetwork_loss_type loss_calc_type = uNNetwork_loss_type::MSE, size_t NLL_target_neuron = 0){

            // calculate losses
            uValue total_loss;
            if(loss_calc_type == uNNetwork_loss_type::MSE){
                // mean squarred error
                for(size_t i = 0; i < output_layer.neuron_count; i++){
                    uValue diff = (uValue(target[i]) - output[i]);
                    total_loss += diff * diff;
                }
            }else if(loss_calc_type == uNNetwork_loss_type::NLL){
                // negetive log likelihood
                for(size_t i = 0; i < output_layer.neuron_count; i++){
                    if(i == NLL_target_neuron){
                        total_loss += (output[i].log() * -1.0f);
                    }else{
                        uValue one(1.0f);
                        total_loss += ((one - output[i]).log() * -1.0f);
                    }
                }
            }else{
                std::cout << "ERROR: Unknown loss calculation type!\n";
            }
            total_loss.backward();

            // propagate backward (output layer)
            output_layer.train(learning_rate);

            // propagate backward (hidden layers)
            for(size_t k = 0; k < hidden_layers.size(); k++){
                hidden_layers[k].train(learning_rate);
            }

            // propagate backward (input layer)
            input_layer.train(learning_rate);

        }


};

struct bigram{
    uint8_t key;
    uint8_t target;
};

#define START_TOK (27ULL)
#define END_TOK (26ULL)

std::vector<bigram> bigrams;

void read_bigrams(const char* path){
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    size_t len = 0;
    uint8_t last = START_TOK;
    for(size_t i = 0; i < content.size(); i++){
        if(content[i] <= 'z' && content[i] >= 'a'){
            len++;

            bigram bg;
            bg.key = last;
            bg.target = content[i] - 'a';
            bigrams.push_back(bg);
        
            last = content[i] - 'a';
        }else if(content[i] == '\n'){
            if(len != 0){
                bigram bg;
                bg.key = last;
                bg.target = END_TOK;
                bigrams.push_back(bg);
            }
            len = 0;
            last = START_TOK;
        }
    }
}

int main(){
    uNodePool_expand_pool(UValueNodePool_chunk_allocate_count);

    read_bigrams("names.txt");

    std::cout << "bigrams read!\n";

    /*
    for(int i= 0; i < bigrams.size(); i++){
        std::cout << i << ". {" << (int)bigrams[i].key << ", " << (int)bigrams[i].target << "}\n";
    }
    */

    // setup
    uLayer n = uLayer::create_mlp_layer(28, 28);

    for(size_t i = 0; i < bigrams.size(); i++){
        uNodePool_temp_begin();

        uValue input[28] = {};

        input[bigrams[i].key] = 1.0f;

        std::vector<uValue> result = n.forward_layer(input); // 28 input

        uValue loss = (result[bigrams[i].target].log() * -1.0f);

        n.train(0.08f);

        uValue::reset_uValue_backward_pool(); // reset pool for next cycle

        uNodePool_temp_end();

        //if((((bigrams.size() * 100) / i) % 10) == 0){
            //std::cout << "INFO: " << ((bigrams.size() * 100) / i) << "% trained...\n";
        //}
    }

    std::cout << "INFO: training completed!\n";

    // generate words
    size_t generated = 0;
    uint8_t last = START_TOK;
    while (generated < 5)
    {
        uNodePool_temp_begin();

        uValue input[28] = {};
        input[last] = 1.0f;

        std::vector<uValue> result = n.forward_layer(input); // 4 input

        uint8_t biggest = 0;
        for(size_t i = 0; i < result.size(); i++){
            if(result[i].get_value() >= result[biggest].get_value()){
                biggest = i;
            }
        }

        if(biggest == END_TOK){
            last = START_TOK;
            std::cout << "\n";

        }else{
            std::cout << (char)(biggest + 'a');
        }

        uNodePool_temp_end();

    }

    // program end
    uValue::flush_uValue_pool();
    return 0;
}

