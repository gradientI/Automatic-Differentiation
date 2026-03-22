#include "autograd.h"
#include <vector>
using namespace std;

// Vector to store intermediate Nodes
vector<Node*> intermediate_nodes;

// Overloading operator to create a new node and calculate local_grad
Node& operator+(Node& a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value + b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);
    newnode->parents.push_back(&b);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(1); // del(newnode) / del a
    newnode->local_grad.push_back(1); // // del(newnode) / del b

    intermediate_nodes.push_back(newnode);

    return *newnode; // return the new node struct
}

Node& operator-(Node& a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value - b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);
    newnode->parents.push_back(&b);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(1); // del(newnode) / del a
    newnode->local_grad.push_back(-1); // // del(newnode) / del b

    intermediate_nodes.push_back(newnode);
    return *newnode;
}
Node& operator-(Node& a, float b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value - b;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(1); // del(newnode) / del a

    intermediate_nodes.push_back(newnode);
    return *newnode;
}
Node& operator-(float a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a - b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&b);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(-1);

    intermediate_nodes.push_back(newnode);
    return *newnode;
}

Node& operator*(Node& a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value * b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);
    newnode->parents.push_back(&b);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(b.value); // del(newnode) / del a
    newnode->local_grad.push_back(a.value); // del(newnode) / del b

    intermediate_nodes.push_back(newnode);

    return *newnode;
}
Node& operator*(float a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a * b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&b);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(a);

    intermediate_nodes.push_back(newnode);

    return *newnode;
}
Node& operator*(Node& a, float b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value * b;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);

    // Storing local grad of newnode in its struct
    newnode->local_grad.push_back(b);

    intermediate_nodes.push_back(newnode);

    return *newnode;
}

Node& operator/(Node& a, Node& b){

    // Creating a new Node to store this operation 
    Node* newnode = new Node;

    newnode->value = a.value / b.value;

    // Adding parents as a and b in new node's parents
    newnode->parents.push_back(&a);
    newnode->parents.push_back(&b);

    newnode->local_grad.push_back(1/b.value); // del(newnode) / del a
    newnode->local_grad.push_back(((-1)*a.value)/(b.value * b.value)); // del(newnode) / del b

    intermediate_nodes.push_back(newnode);
    
    return *newnode;
}

// Backward function which will trace graph backwards and calculate derivative
// Input will be the last node 
void backward(Node* &last_node){

    // Condition to stop recursion 
    if (last_node->parents.empty())
    {
        return;
    }

    // Push the parent nodes of last node into stack
    // And calculate global grad
    for(int i = 0; i < last_node->parents.size(); i++){
        last_node->parents[i]->global_grad += last_node->global_grad * last_node->local_grad[i];
        backward(last_node->parents[i]);
    }
}

// Function to create a weight matrix
vector<vector<Node*>> weight_matrix(int rows, int cols){

    vector<vector<Node*>> W(rows, vector<Node*>(cols));

    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            W[i][j] = new Node(0);
        }
    }
    return W;
}

// Function to create a weight vector
vector<Node*> weight_vector(int size){

    vector<Node*> W(size);

    for(int i = 0; i<size; i++){
        W[i] = new Node(0);
    }
    return W;
}


// Function to do a vector matrix product
vector<Node*> vec_mat_product(vector<float>& vec, vector<vector<Node*>>& mat){
    
    int mat_rows = mat.size();
    int mat_cols = mat[0].size();
    
    vector<Node*> result(mat_cols);

    for(int j = 0; j < mat_cols; j++){
        result[j] = new Node(0);  // initialize to 0
        for(int i = 0; i < mat_rows; i++){
            Node* scaled = &((*mat[i][j]) * vec[i]);
            result[j] = &((*result[j]) + *scaled);
        }
    }
    return result;
}

vector<Node*> vec_mat_product(vector<Node*>& vec, vector<vector<Node*>>& mat){
    int mat_rows = mat.size();
    int mat_cols = mat[0].size();
    
    vector<Node*> result(mat_cols);

    for(int j = 0; j < mat_cols; j++){
        result[j] = new Node(1); 

        for(int i = 0; i < mat_rows; i++){
            result[j] = &(*result[j] + ((*mat[i][j]) * *vec[i]));
        }
    }
    return result;
}

// Function to do a vector vector addition
vector<Node*> vec_addition(vector<Node*>& a, vector<Node*>& b){

    vector<Node*> r(a.size());
    for(int i = 0; i<a.size(); i++){

        r[i] = &(*a[i] + *b[i]);
    }

    return r;
}

// ReLU function
Node& relu(Node& a){

    Node* newnode = new Node;

    if(a.value <= 0){
        newnode->value = 0;
        newnode->local_grad.push_back(0);
    }else{
        newnode->value = a.value;
        newnode->local_grad.push_back(1);
    }

    newnode->parents.push_back(&a);
    intermediate_nodes.push_back(newnode);

    return *newnode;
}

// Matrix scalar product
void scale_matrix(float scalar, vector<vector<Node*>>& mat){

    for(int i = 0; i<mat.size(); i++){
        for(int j = 0; j<mat[0].size(); j++){
            mat[i][j]->value = mat[i][j]->value * scalar;
        }
    }
}

// vector scalar product
void scale_vector(float scalar, vector<Node*>& vec){

    for(int i = 0; i<vec.size(); i++){
        vec[i]->value = vec[i]->value * scalar;
    }
}

// Update matrix
void update_weight_matrix(vector<vector<Node*>>& mat, float alpha){

    for(int i = 0; i<mat.size(); i++){
        for(int j = 0; j<mat[0].size(); j++){
            mat[i][j]->value = mat[i][j]->value - (alpha * (mat[i][j]->global_grad));
            mat[i][j]->global_grad = 0;
            mat[i][j]->parents.clear();
        }
    }
}

// Update Bias vector
void update_weight_vector(vector<Node*>& vec, float alpha){

    for(int i = 0; i < vec.size(); i++){
        vec[i]->value = vec[i]->value - (alpha * vec[i]->global_grad);
        vec[i]->global_grad = 0;
        vec[i]->parents.clear();
    }
}



Neuron::Neuron(int numinp){
    for(int i = 0; i < numinp; i++){
        weights.push_back(new Node(0));
    }
    bias = new Node(0);
}

    // forward pass 
Node* Neuron::forward(vector<float>& inputs, bool act_func){
    Node* result = &(*weights[0] * inputs[0]);
    for(int i = 1; i < inputs.size(); i++){
        Node* scaled = &(*weights[i] * inputs[i]);
        result = &(*result + *scaled);
    }
    if(act_func == true){
        result = &relu(*result + *bias);
        return result;
    }else{
        result = &(*result + *bias);
        return result;
    }
}

    // forward pass 
Node* Neuron::forward(vector<Node*>& inputs, bool act_func){
    Node* result = &(*weights[0] * *inputs[0]);

    for(int i = 1; i < inputs.size(); i++){
        Node* scaled = &(*weights[i] * *inputs[i]);
        result = &(*result + *scaled);
    }
    if(act_func == true){
        result = &relu(*result + *bias);
        return result;
    }else{
        result = &(*result + *bias);
        return result;
    }
}


Layer::Layer(int numip, int numop){
    for(int i = 0; i<numop; i++){
        neurons.push_back(Neuron(numip));
    }
}

vector<Node*> Layer::forward(vector<float>& inputs, bool act_func){
    
    vector<Node*> out;

    for(int i = 0; i<neurons.size(); i++){
        out.push_back(neurons[i].forward(inputs, act_func));
    }
    return out;
}

vector<Node*> Layer::forward(vector<Node*>& inputs, bool act_func){
    
    vector<Node*> out;

    for(int i = 0; i<neurons.size(); i++){
        out.push_back(neurons[i].forward(inputs, act_func));
    }
    return out;
    }