#include "autograd.h"
using namespace std;
#include <iostream>
#include <vector>

float alpha = 0.0001; // Learning Rate

int main(){

    // Input output pair
    vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<int> output = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // function will be y = wx + b

    Node w = 23.21; // weight randomly initialized
    Node b = 3.2323;

    // Running loop e times, each inner loop takes one example to update weights

    for(int e = 0; e < 5000; e++){

        for(int i = 0; i<10; i++){

            Node y = ((w * input[i]) + b); //Forward pass

            Node l = (output[i] - y) * (output[i] - y); //Loss

            Node* ln = &l;
            ln->global_grad = 1;
            backward(ln); // Calling backward on l (loss)

            // updating weight and bias

            w.value = w.value - (alpha * w.global_grad);
            b.value = b.value - (alpha * b.global_grad);

            // Reseting gradients
            w.global_grad = 0;
            b.global_grad = 0;

            // Clearing 
            intermediate_nodes.clear();

            w.parents.clear();
            b.parents.clear();
            l.parents.clear();
            y.parents.clear();
        }
    }
    
    cout<<w.value<<endl;
    return 0;
}