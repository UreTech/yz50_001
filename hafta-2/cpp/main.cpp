#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numbers>
#include <random>

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

struct operation_data
{
        uValueNode* left = nullptr;
        uValueNode* right = nullptr;
        operation_type op_type = operation_type::UNDEFINED;
};

std::vector<void*> ptrs;

class uValueNode{
    public:
        std::vector<operation_data> operations;
        float value = 0.0f;
        float grad = 0.0f;

        uValueNode* create_copy(){
            uValueNode* copy = new uValueNode(*this);
            copy->grad = 0.0f;
            return copy;
        }

        void* operator new(size_t size){
            void* ptr = malloc(size);
            ptrs.push_back(ptr);
            return ptr;
        }

        void operator delete(void* ptr){
            // do nothing
            // pool managed manualy
        }
};  

class uValue{
    private:
        uValueNode *node = nullptr;
    public:


        uValue(float val){
            node = new uValueNode();
            this->node->value = val;
            this->node->grad = 0.0f;
        }

        uValue(){
            node = new uValueNode();
            this->node->value = 0.0f;
            this->node->grad = 0.0f;
        }

        // initalize with existing node
        uValue(uValueNode* node_){
            node = node_;
        }


        static void flush_uValue_pool(){
            for(size_t i = 0; i < ptrs.size(); i++){
                uValueNode* node_ = (uValueNode*)ptrs[i];
                node_->operations.clear();
                node_->~uValueNode();
                free(ptrs[i]);
            }
            ptrs.clear();
        }

        static void reset_uValue_backward_pool(){
            for(size_t i = 0; i < ptrs.size(); i++){
                uValueNode* node_ = (uValueNode*)ptrs[i];
                node_->grad = 0.0f;
                node_->operations.clear(); // clear operations
            }
        }

        float get_value(){
            return this->node->value;
        }

        float get_grad(){
            return this->node->grad;
        }

        uValue tanh(){

            // tanh formula
            // NOTE: this formula is not  optimised  for float. IT OVER FLOWS!
            // maybe clamping output to betweeen -1 & 1 solves problem but who cares
            // uValue child((std::pow(M_E, this->node->value) - std::pow(M_E, -(this->node->value))) / (std::pow(M_E, this->node->value) + std::pow(M_E, -(this->node->value))));

            uValue child(std::tanh(this->node->value));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = nullptr;
            op.op_type = operation_type::TANH;

            child.node->operations.push_back(op);

            return child;
        }

        uValue pow(uValue power){

            uValue child(std::pow(this->node->value, power.node->value));

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = power.node;
            op.op_type = operation_type::POW;

            child.node->operations.push_back(op);

            return child;
        }


        uValue operator+(const uValue& other){
            uValue child(this->node->value + other.node->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::ADD;

            child.node->operations.push_back(op);

            return child;
        }

        uValue& operator+=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = this->node->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::ADD;

            this->node->value += other.node->value;

            this->node->operations.clear(); // old history holded by copy so we should clear this history for preventing double referance
            this->node->operations.push_back(op);

            return *this;
        }

        uValue operator-(const uValue& other){
            uValue child(this->node->value - other.node->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::SUB;

            child.node->operations.push_back(op);

            return child;
        }

        uValue& operator-=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = this->node->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::SUB;

            this->node->value -= other.node->value;

            this->node->operations.clear(); // old history holded by copy so we should clear this history for preventing double referance
            this->node->operations.push_back(op);

            return *this;
        }

        uValue operator*(const uValue& other){
            uValue child(this->node->value * other.node->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::MUL;

            child.node->operations.push_back(op);

            return child;
        }

        uValue& operator*=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = this->node->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::MUL;

            this->node->value *= other.node->value;

            this->node->operations.clear(); // old history holded by copy so we should clear this history for preventing double referance
            this->node->operations.push_back(op);

            return *this;
        }

        uValue operator/(const uValue& other){
            uValue child(this->node->value / other.node->value);

            // left & right
            operation_data op = {};
            op.left = this->node;
            op.right = other.node;
            op.op_type = operation_type::DIV;

            child.node->operations.push_back(op);

            return child;
        }

        uValue& operator/=(const uValue& other){
            // left & right
            operation_data op = {};
            op.left = this->node->create_copy(); // copy needed because we dont want to destroy or lose history of value
            op.right = other.node;
            op.op_type = operation_type::DIV;

            this->node->value /= other.node->value;

            this->node->operations.clear(); // old history holded by copy so we should clear this history for preventing double referance
            this->node->operations.push_back(op);

            return *this;
        }

        uValue& operator=(const uValue& other){
            // this could cause a dangling non refernced node but our garbage collector should free it when needed
            this->node = other.node;
            return *this;
        }

        // float is out side of our scope
        uValue& operator=(const float other){
            this->node->value = other;
            this->node->grad = 0.0f;
            this->node->operations.clear();
            return *this;
        }

        std::string dump_parents(){
            std::string result = "";
            if(this->node->operations.empty()){
                result += "{[val]: " + std::to_string(this->node->value) + " [grad]: " + std::to_string(this->node->grad) + "}";
            }

            for(size_t i = 0; i < this->node->operations.size(); i++){
                switch (this->node->operations[i].op_type)
                {
                case ADD:
                    result += "(" + uValue(this->node->operations[i].left).dump_parents() + " + " + uValue(this->node->operations[i].right).dump_parents() + ")";
                    break;  

                case SUB:
                    result += "(" + uValue(this->node->operations[i].left).dump_parents() + " - " + uValue(this->node->operations[i].right).dump_parents() + ")";
                    break;

                case MUL:
                    result += "(" + uValue(this->node->operations[i].left).dump_parents() + " * " + uValue(this->node->operations[i].right).dump_parents() + ")";
                    break;

                case DIV:
                    result += "(" + uValue(this->node->operations[i].left).dump_parents() + " / " + uValue(this->node->operations[i].right).dump_parents() + ")";
                    break;

                case POW:
                    result += ("pow(" + uValue(this->node->operations[i].left).dump_parents() + " ," + uValue(this->node->operations[i].right).dump_parents() + ")");
                    break;                    

                case TANH:
                    result += ("tanh(" + uValue(this->node->operations[i].left).dump_parents() + ")");
                    break;
                
                default:
                    break;
                }
            }

            return result;
        }

        // backward gradient calculation
        void backward(bool root = true){

            if(root){
                this->node->grad = 1.0f;
            }

            for(size_t i = 0; i < this->node->operations.size(); i++){
                // calculate gradians
                switch (this->node->operations[i].op_type)
                {
                case ADD:
                    this->node->operations[i].left->grad += this->node->grad;
                    this->node->operations[i].right->grad += this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    uValue(this->node->operations[i].right).backward(false);
                    break;  

                case SUB:
                    this->node->operations[i].left->grad += this->node->grad;
                    this->node->operations[i].right->grad -= this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    uValue(this->node->operations[i].right).backward(false);
                    break;

                case MUL:
                    this->node->operations[i].left->grad += this->node->operations[i].right->value * this->node->grad;
                    this->node->operations[i].right->grad += this->node->operations[i].left->value * this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    uValue(this->node->operations[i].right).backward(false);
                    break;

                case DIV:
                    this->node->operations[i].left->grad += (1.0f / this->node->operations[i].right->value) * this->node->grad;
                    this->node->operations[i].right->grad += (-(this->node->operations[i].left->value) / std::pow(this->node->operations[i].right->value, 2)) * this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    uValue(this->node->operations[i].right).backward(false);
                    break;

                case TANH:
                    this->node->operations[i].left->grad += (1.0f - std::pow(this->node->value, 2)) * this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    break;

                case POW:
                    this->node->operations[i].left->grad += this->node->operations[i].right->value * std::pow(this->node->operations[i].left->value, this->node->operations[i].right->value - 1.0f) * this->node->grad;
                    this->node->operations[i].right->grad += this->node->value * std::log(this->node->operations[i].left->value) * this->node->grad;
                    uValue(this->node->operations[i].left).backward(false);
                    uValue(this->node->operations[i].right).backward(false);
                    break;
                
                default:
                    break;
                }
            }
        }
    
};

class uNeuron{
    public:
        std::vector<uValue> weights;
        uValue bias = 1.0f;
        static uNeuron create_neuron(size_t input_count){
            uNeuron neuron;
            neuron.weights.resize(input_count);
            for(size_t i = 0; i < neuron.weights.size(); i++){
                neuron.weights[i] = random_float(-1.0f, 1.0f);
            }
            neuron.bias = random_float(-1.0f, 1.0f);
            return neuron;
        }
};

class uLayer{
    public:
        size_t input_count;
        std::vector<uNeuron> neurons;

        static uLayer create_layer(size_t input_count, size_t neuron_count){
            uLayer layer;
            layer.input_count = input_count;
            layer.neurons.resize(neuron_count);

            for(size_t i = 0; i < neuron_count; i++){
                layer.neurons[i] = uNeuron::create_neuron(input_count);
            }

            return layer;
        }

        std::vector<uValue> forward_layer(uValue* input){
            std::vector<uValue> output;
            for(size_t i = 0; i < this->neurons.size(); i++){
                // calculate neuron output
                uValue y = 0.0f;
                for(size_t j = 0; j < this->neurons[i].weights.size(); j++){
                    y += (this->neurons[i].weights[j] * input[j]);
                }
                y += this->neurons[i].bias;
                output.push_back(y.tanh());
            }
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
            for(size_t i = 0; i < output_layer.neurons.size(); i++){
                uValue diff = (uValue(target[i]) - output[i]);
                total_loss += diff * diff;
            }
            total_loss.backward();

            // propagate backward (output layer)
            for(size_t i = 0; i < output_layer.neurons.size(); i++){
                for(size_t j = 0; j < output_layer.neurons[i].weights.size(); j++){
                    // -= is not used because we dont want to keep history
                    output_layer.neurons[i].weights[j] = (output_layer.neurons[i].weights[j] - (output_layer.neurons[i].weights[j].get_grad() * learning_rate)).get_value();
                }
                output_layer.neurons[i].bias = (output_layer.neurons[i].bias - (output_layer.neurons[i].bias.get_grad() * learning_rate)).get_value();
            }

            // propagate backward (hidden layers)
            for(size_t k = 0; k < hidden_layers.size(); k++){
                for(size_t i = 0; i < hidden_layers[k].neurons.size(); i++){
                    for(size_t j = 0; j < hidden_layers[k].neurons[i].weights.size(); j++){
                        // -= is not used because we dont want to keep history
                        hidden_layers[k].neurons[i].weights[j] = (hidden_layers[k].neurons[i].weights[j] - (hidden_layers[k].neurons[i].weights[j].get_grad() * learning_rate)).get_value();
                    }
                    hidden_layers[k].neurons[i].bias = (hidden_layers[k].neurons[i].bias - (hidden_layers[k].neurons[i].bias.get_grad() * learning_rate)).get_value();
                }
            }

            // propagate backward (input layer)
            for(size_t i = 0; i < input_layer.neurons.size(); i++){
                for(size_t j = 0; j < input_layer.neurons[i].weights.size(); j++){
                    // -= is not used because we dont want to keep history
                    input_layer.neurons[i].weights[j] = (input_layer.neurons[i].weights[j] - (input_layer.neurons[i].weights[j].get_grad() * learning_rate)).get_value();
                }
                input_layer.neurons[i].bias = (input_layer.neurons[i].bias - (input_layer.neurons[i].bias.get_grad() * learning_rate)).get_value();
            }

        }


};

int main(){

    // setup mlp
    uMLP mlp;
    mlp.input_layer = uLayer::create_layer(4, 3); // 4 input & 3 neurons
    uLayer hidden0 = uLayer::create_layer(3, 16); // 3 input & 16 neuron
    mlp.hidden_layers.push_back(hidden0);
    mlp.output_layer = uLayer::create_layer(16, 2); // 16 input & 2 neurons
    
    float inputs[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float targets[] = {0.03f, 0.01f};

    size_t training_cycles = 1500;
    while(training_cycles--){
        std::vector<uValue> result = mlp.forward(inputs); // 4 input
        mlp.backward_training(targets, result.data(), 0.0001f); // 2 output
        uValue::reset_uValue_backward_pool(); // reset pool for next cycle
    }

    std::vector<uValue> result = mlp.forward(inputs); // 4 input

    std::cout << "trained:\n";
    for(size_t i = 0; i < result.size(); i++){
        std::cout << i << ": " << result[i].get_value() << "\n";
    }

    // program end
    uValue::flush_uValue_pool();
    return 0;
}

