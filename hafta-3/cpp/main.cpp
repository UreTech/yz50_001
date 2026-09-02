#include <hip/hip_runtime.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numbers>
#include <random>
#include <cstdio>

#define UNODE_DEVICE_BUFFER_ENABLE_OVERFLOW_WARNINGS true

#define UValueNodePool_chunk_allocate_count (512ULL)
#define UValueNodePool_gpu_chunk_allocate_count (2048ULL * 16ULL) // larger for not wasting gpu

float random_float(float min, float max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(min, max);

    return dist(gen);
}

typedef enum{
    ADD,
    SUB,
    DIV,
    MUL,
    TANH,
    POW,
    UNDEFINED,
}operation_type;

// forward decs
class uValue;
class uValueNode;

#define INVALID_NODE 0ULL

struct uNodePool{
    std::vector<uValueNode> uValue_pool;
    uint64_t next = 0; // last available node
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
        uNodeIdx left = UINT64_MAX;
        uNodeIdx right = UINT64_MAX;
        operation_type op_type = operation_type::UNDEFINED;
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
                atomicExch(&(gpu_pool.overflowed), true);
                atomicMin(&gpu_pool.overflow_index, blockIdx.x * blockDim.x + threadIdx.x);
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
            return &(gpu_pool.uValue_pool[idx]);
        }

};  

bool uNodePool_copy_device_to_host(){
    // firstly get device pool
    hipMemcpyFromSymbol(&gpu_pool_host_copy, HIP_SYMBOL(gpu_pool), sizeof(gpu_uNodePool), 0, hipMemcpyDeviceToHost);

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
    hipMemcpy(pool.uValue_pool.data(), gpu_pool_host_copy.uValue_pool, sizeof(uValueNode) * pool.uValue_pool.size(), hipMemcpyDeviceToHost);

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
                hipMalloc(&gpu_pool_host_copy.uValue_pool, gpu_pool_host_copy.pool_size * sizeof(uValueNode));
            }else{
                hipFree(gpu_pool_host_copy.uValue_pool);
                gpu_pool_host_copy.uValue_pool = nullptr;
                hipMalloc(&gpu_pool_host_copy.uValue_pool, gpu_pool_host_copy.pool_size * sizeof(uValueNode));
            }

        }

        if(gpu_pool_host_copy.uValue_pool == nullptr){
            std::cout << "ERROR: Failed to allocate gpu node table!\n";
        }
    }else{
        hipFree(gpu_pool_host_copy.uValue_pool);
        gpu_pool_host_copy.pool_size = 0;
        gpu_pool_host_copy.uValue_pool = nullptr;
    }

    gpu_pool_host_copy.next = pool.next;

    // then copy pool content in to device pool
    hipMemcpy(gpu_pool_host_copy.uValue_pool, pool.uValue_pool.data(), sizeof(uValueNode) * gpu_pool_host_copy.pool_size, hipMemcpyHostToDevice);

    // finaly copy host pool to device
    hipMemcpyToSymbol(HIP_SYMBOL(gpu_pool), &gpu_pool_host_copy, sizeof(gpu_uNodePool), 0, hipMemcpyHostToDevice);

    // done!
}

class uValue{
    public:

        uNodeIdx node = UINT64_MAX;

        __host__ __device__
        uValue(float val){
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
        __host__ __device__
        uValue(uNodeIdx node_, int _this_arg_is_for_tricking_compiler_){
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
            op.right = UINT64_MAX;
            op.op_type = operation_type::TANH;

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
            if(uValueNode::get_node(this->node)->parent_operation.op_type == operation_type::UNDEFINED){
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

            case TANH:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += (1.0f - std::pow(uValueNode::get_node(this->node)->value, 2)) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                break;

            case POW:
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->grad += uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value * ::pow(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value, uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->value - 1.0f) * uValueNode::get_node(this->node)->grad;
                uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.right)->grad += uValueNode::get_node(this->node)->value * ::log(uValueNode::get_node(uValueNode::get_node(this->node)->parent_operation.left)->value) * uValueNode::get_node(this->node)->grad;
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.left).backward(false);
                uValue::from_node(uValueNode::get_node(this->node)->parent_operation.right).backward(false);
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
            hipMalloc(&(neuron.weights_device_buffer), neuron.weight_count * sizeof(uValue));

            if(neuron.weights_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu neuron buffer!\n";
            }

            for(size_t i = 0; i < neuron.weight_count; i++){
                neuron.weights[i] = random_float(-0.3f, 0.7f);
            }
            neuron.bias = random_float(-1.0f, 1.0f);
            return neuron;
        }

        void copy_neuron_to_device(){
            hipMemcpy(weights_device_buffer, weights, sizeof(uValue) * weight_count, hipMemcpyHostToDevice);
        }

        void copy_neuron_to_host(){
            hipMemcpy(weights, weights_device_buffer, sizeof(uValue) * weight_count, hipMemcpyDeviceToHost);
        }
};

__global__
void _forward_layer_gpu_thread_(uValue* input, uValue output_start, uNeuron* neurons, size_t neuron_count){
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < neuron_count){
        // calculate neuron output
        uValue y = 0.0f;
        for(size_t j = 0; j < neurons[id].weight_count; j++){
            y += (neurons[id].weights_device_buffer[j] * input[id]);
        }
        y += neurons[id].bias;
        uValue::from_node(output_start.node + id) = y.tanh();
    }
}

__global__
void _train_neuron_gpu_thread_(float learning_rate, uNeuron* neurons, size_t neuron_count){
    uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < neuron_count){
        for(size_t j = 0; j < neurons[id].weight_count; j++){
            neurons[id].weights_device_buffer[j] = (neurons[id].weights_device_buffer[j] - (neurons[id].weights_device_buffer[j].get_grad() * learning_rate)).get_value();
        }
        neurons[id].bias = (neurons[id].bias - (neurons[id].bias.get_grad() * learning_rate)).get_value();
    }
} 

class uLayer{
    public:
        size_t input_count;
        size_t neuron_count;
        uNeuron* neurons = nullptr;
        uNeuron* neurons_device_buffer = nullptr;

        static uLayer create_layer(size_t input_count, size_t neuron_count){
            uLayer layer;
            layer.input_count = input_count;
            layer.neuron_count = neuron_count;
            layer.neurons = new uNeuron[layer.neuron_count];

            hipMalloc(&(layer.neurons_device_buffer), neuron_count * sizeof(uNeuron));

            if(layer.neurons_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu neurons buffer!\n";
            }

            for(size_t i = 0; i < neuron_count; i++){
                layer.neurons[i] = uNeuron::create_neuron(input_count);
            }

            return layer;
        }

        void copy_neurons_to_device(){
            for(size_t i = 0; i < neuron_count; i++){
                neurons[i].copy_neuron_to_device();
            }
            hipMemcpy(neurons_device_buffer, neurons, neuron_count * sizeof(uNeuron), hipMemcpyHostToDevice);
        }

        void copy_neurons_to_host(){
            for(size_t i = 0; i < neuron_count; i++){
                neurons[i].copy_neuron_to_host();
            }
            hipMemcpy(neurons, neurons_device_buffer, neuron_count * sizeof(uNeuron), hipMemcpyDeviceToHost);
        }

        __host__
        void train(float learning_rate){
            copy_neurons_to_device();
            
            while(true){
                uNodePool_copy_host_to_device();

                size_t threads_per_block = 256;
                size_t blocks = (neuron_count + threads_per_block - 1) / threads_per_block;

                hipLaunchKernelGGL(_train_neuron_gpu_thread_, dim3(blocks), dim3(threads_per_block), 0, 0, learning_rate, this->neurons_device_buffer, this->neuron_count);

                hipDeviceSynchronize();

                if(uNodePool_copy_device_to_host()){
                    copy_neurons_to_host();
                    break;
                }
            }
        }

        std::vector<uValue> forward_layer(uValue* input){
            std::vector<uValue> output(this->neuron_count); // output

            // input
            uValue* input_device_buffer = nullptr;
            hipMalloc(&input_device_buffer, sizeof(uValue) * this->input_count);
            if(input_device_buffer == nullptr){
                std::cout << "ERROR: Failed to allocate gpu input device buffer!\n";
                return output;
            }
            hipMemcpy(input_device_buffer, input, sizeof(uValue) * this->input_count, hipMemcpyHostToDevice);
            
            copy_neurons_to_device();

            size_t threads_per_block = 256;
            size_t blocks = (neuron_count + threads_per_block - 1) / threads_per_block;

            while(true){

                // now copy to device
                uNodePool_copy_host_to_device();

                hipLaunchKernelGGL(_forward_layer_gpu_thread_, dim3(blocks), dim3(threads_per_block), 0, 0, input_device_buffer, output[0], this->neurons_device_buffer, this->neuron_count);

                hipDeviceSynchronize();

                // read back to host
                if(uNodePool_copy_device_to_host() == true){
                    copy_neurons_to_host();
                    break;
                }
                #if UNODE_DEVICE_BUFFER_ENABLE_OVERFLOW_WARNINGS
                std::cout << "WARN! Forward failed because of overflow!\n";
                #endif
            }

            hipFree(input_device_buffer);
            return output;
        }
};

class uMLP{
    public:
        uLayer input_layer;
        std::vector<uLayer> hidden_layers;
        uLayer output_layer;

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

        void backward_training(float* target, uValue* output, float learning_rate){

            // calculate losses
            uValue total_loss;
            for(size_t i = 0; i < output_layer.neuron_count; i++){
                uValue diff = (uValue(target[i]) - output[i]);
                total_loss += diff * diff;
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

int main(){

    // setup mlp
    uMLP mlp;
    mlp.input_layer = uLayer::create_layer(4, 3); // 4 input & 3 neurons
    uLayer hidden0 = uLayer::create_layer(3, 4); // 3 input & 16 neuron
    mlp.hidden_layers.push_back(hidden0);
    mlp.output_layer = uLayer::create_layer(4, 2); // 16 input & 2 neurons
    
    float inputs1[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float targets1[] = {-1.0f, 1.0f};

    float inputs2[] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float targets2[] = {1.0f, -1.0f};

    size_t training_cycles = 4000;
    bool gio = false;
    while(training_cycles--){
        float *inputs;
        float *targets;
        if(gio){
            inputs = inputs1;
            targets = targets1;
        }else{
            inputs = inputs2;
            targets = targets2;
        }
        gio = !gio;

        uNodePool_temp_begin();
        std::vector<uValue> result = mlp.forward(inputs); // 4 input
        mlp.backward_training(targets, result.data(), 0.08f); // 2 output
        uValue::reset_uValue_backward_pool(); // reset pool for next cycle
        uNodePool_temp_end();
    }

    std::vector<uValue> result = mlp.forward(inputs1); // 4 input

    std::cout << "trained1:\n";
    for(size_t i = 0; i < result.size(); i++){
        std::cout << i << ": " << result[i].get_value() << "\n";
    }

    result = mlp.forward(inputs2); // 4 input

    std::cout << "trained2:\n";
    for(size_t i = 0; i < result.size(); i++){
        std::cout << i << ": " << result[i].get_value() << "\n";
    }

    // program end
    uValue::flush_uValue_pool();
    return 0;
}

