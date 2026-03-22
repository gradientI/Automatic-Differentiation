# Automatic-Differentiation
A "Non-working" automatic differentiation library (forward and reverse mode) with unoptimized recursive implementation

In the forward sweep the local gradients are calculated and intermediate nodes are created, the nodes which created the intermediate node are parents of the intermediate node and so on.
during backward pass the global grad of last node is set to 1 and then recursively goes and applies the chain rule i.e for node n, del n / del (last_node) is sum of n's child node's global grad * del (n's child node) / del (n)
i.e if n has two children a and b then n.global_grad = (a.global_grad * del a / del n) + (b.global_grad * del b / del n)
there is some deep issue in this code, this was for learning purposes and i am stuck, if anyone can help or fix this then that would be helpfull, thank you !!
