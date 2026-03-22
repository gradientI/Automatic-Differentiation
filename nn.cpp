#include "autograd.h"
#include <vector>
#include <iostream>
#include <random>

using namespace std;

// Creating weight matrixes
// Two layers
// ip vec size = ip, hidden units in layer 1 => HU1, hidden units in layer 1 => HU2,
// Output of size = op 

int ipsize = 6;
int HU1 = 64;
int HU2 = 64;
int opsize = 1;

float alpha = 0.0001;

// Demo input, each row is one example input, 6 variables
int numexamples = 100; // 100 total examples
// output is a function f(X), X is R_6
// f(X) = (x1*x2) + ((x1*x1)/(x5+x3+1e-8)) + (x6*(x4-x6)) + (x1*(x3-x4))

pair<vector<vector<float>>, vector<float>> generate_data(int numexamples){

    vector<vector<float>> input(numexamples, vector<float>(6));
    vector<float> output(numexamples);

    mt19937 gen(42);
    uniform_real_distribution<float> dis(-1.0, 1.0);

    for(int i = 0; i < numexamples; i++){
        for(int j = 0; j < 6; j++){
            input[i][j] = dis(gen);
        }
    }

    for(int i = 0; i < numexamples; i++){
        float x1 = input[i][0];
        float x2 = input[i][1];
        float x3 = input[i][2];
        float x4 = input[i][3];
        float x5 = input[i][4];
        float x6 = input[i][5];

        output[i] = (x1*x2) + ((x1*x1)/(x5+x3+1e-8)) + (x6*(x4-x6)) + (x1*(x3-x4));
    }

    return {input, output};
}

vector<vector<Node*>> W1 = weight_matrix(ipsize, HU1);
vector<Node*> B1 = weight_vector(HU1);

vector<vector<Node*>> W2 = weight_matrix(HU1, opsize);
vector<Node*> B2 = weight_vector(opsize);

/*
// Model for inference, testing
float model(vector<float>& ip){
            // l1 = (W1 * ip) + B1

            vector<Node*> l1 = vec_mat_product(ip, W1);
            l1 = vec_addition(l1, B1);

            // Applying relu to l1
            vector<Node*> a1(l1.size());
            for(int j = 0; j<l1.size(); j++){
                a1[j] = &relu(*l1[j]);
            }

            // l2 = (W2 * a1) + B2

            vector<Node*> l2 = vec_mat_product(a1, W2);
            l2 = vec_addition(l2, B2);

            // Applying relu to l2
            vector<Node*> a2(l2.size());
            for(int j = 0; j<l2.size(); j++){
                a2[j] = &relu(*l2[j]);
            }

            // op = (W3 * a2) + B3

            vector<Node*> l3 = vec_mat_product(a2, W3);
            l3 = vec_addition(l3, B3);

            return l3[0]->value;
}
*/

Neuron n1(6);

Layer l1(6, 32);
Layer l2(32, 1);

int epochs = 50;

int main(){

    // Neural Network Structure



    // Generating data
    auto [input, output] = generate_data(100);

    cout<<"Data Generated"<<endl;

    for(int e = 0; e<epochs; e++){

        // Average loss over entire dataset
        float avgloss = 0;

        for(int i = 0; i<numexamples; i++){

            vector<float> ip = input[i];
            float op = output[i];


            vector<Node*> out1 = l1.forward(ip, false);
            vector<Node*> out2 = l2.forward(out1, false);
            vector<Node*> prediction = out2;

            Node op_node = op;
            Node& diff = op_node - *prediction[0];
            Node& loss = diff * diff;

            Node* l = &loss;
            l->global_grad = 1;
            backward(l);
            backward(l);

            // Clearing Nodes
            for(Node* n : intermediate_nodes){
                delete n;
            }
            intermediate_nodes.clear();

        }

        avgloss = avgloss / numexamples;
        std::cout<<e+1<<"> "<<"Loss: "<<avgloss<<endl;

    }

// Test
vector<float> test = {1.5342,  // x1
                      4.33,  // x2
                      3.232,  // x3
                      0.83,  // x4
                      2.12312,  // x5
                      1.4,  // x6
};

float x1=test[0], x2=test[1], x3=test[2], x4=test[3];
float x5=test[4], x6=test[5];

float actual = (x1*x2) + ((x1*x1)/(x5+x3+1e-8)) + (x6*(x4-x6)) + (x1*(x3-x4));

//cout << "Predicted: " << model(test) << endl;
cout << "Actual:    " << actual << endl;
}
