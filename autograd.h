#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include <vector>
using namespace std;

struct Node{
    
    float value;
    vector<Node*> parents;  // To store the Node's parents
    float global_grad = 0; // Grad with respect to the last node
    vector<float> local_grad; // Grad wrt node's input 

    Node() {}
    Node(float val) : value(val) {}
}; 

extern vector<Node*> intermediate_nodes;

Node& operator+(Node& a, Node& b);

Node& operator-(Node& a, Node& b);
Node& operator-(Node& a, float b);
Node& operator-(float a, Node& b);

Node& operator*(Node& a, Node& b);
Node& operator*(float a, Node& b);
Node& operator*(Node& a, float b);

Node& operator/(Node& a, Node& b);

// Backward function which will trace graph backwards and calculate derivative
// Input will be the last node 
void backward(Node* &last_node);

// Function to create a weight matrix
vector<vector<Node*>> weight_matrix(int rows, int cols);

//Function to create a weight vector
vector<Node*> weight_vector(int size);

// Function to do a vector matrix product
vector<Node*> vec_mat_product(vector<float>& vec, vector<vector<Node*>>& mat);
vector<Node*> vec_mat_product(vector<Node*>& vec, vector<vector<Node*>>& mat);

// Function to do a vector vector addition
vector<Node*> vec_addition(vector<Node*>& a, vector<Node*>& b);

// ReLU Activation Function
Node& relu(Node& a);

void update_weight_matrix(vector<vector<Node*>>& mat, float alpha);

// Update Bias vector
void update_weight_vector(vector<Node*>& vec, float alpha);

struct Neuron{

    // Takes input and stores the weighted sum of the input
    vector<Node*> weights;
    Node* bias;
    Neuron(int numip);
    Node* forward(vector<float>& inputs, bool act_func);
    Node* forward(vector<Node*>& inputs, bool act_func);

};

struct Layer{

    // Creates a layer object which takes num inputs and num outputs and creates 
    // a num outputs number of neurons
    // During forward pass we input to a layer its previous layer output

    vector<Neuron> neurons;
    Layer(int numip, int numop);
    vector<Node*> forward(vector<float>& inputs, bool act_func);
    vector<Node*> forward(vector<Node*>& inputs, bool act_func);
};

#endif