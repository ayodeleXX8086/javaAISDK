# Tensor Operations Overview

This project implements operations for handling tensors, focusing on `TensorVector` and `TensorScalar` types. Tensors are multi-dimensional arrays used primarily in machine learning and numerical computing. The operations provided in this project allow for efficient manipulation of these tensors, including element-wise operations, broadcasting, slicing, gradient calculation, and SIMD operations for enhanced performance.

## TensorVector

`TensorVector` represents a multi-dimensional array (vector) and provides various methods to perform operations on this array. Key features include:

- **Element-wise Operations**: Supports addition, subtraction, multiplication, and division between `TensorVector` and other tensors, including `TensorScalar`.
- **Broadcasting**: Allows operations between tensors of different shapes by expanding the smaller tensor to match the shape of the larger tensor.
- **Slicing**: Enables extraction of sub-tensors from a larger tensor using specific index ranges.
- **Gradient Calculation**: Supports automatic differentiation, allowing gradients to be computed for each element in the tensor, useful for optimization tasks in machine learning.
- **SIMD Operations**: Utilizes SIMD instructions to perform parallel computations on tensor elements, significantly improving the performance of large-scale tensor operations.

## TensorScalar

`TensorScalar` represents a scalar value (single number) and is used for operations where a tensor interacts with a scalar. Key features include:

- **Element-wise Operations**: Can be added, subtracted, multiplied, or divided with a `TensorVector`, performing the operation element-wise across the tensor.
- **Broadcasting**: Automatically adjusts the scalar to match the shape of the tensor during operations.
- **Gradient Calculation**: Supports backpropagation, computing the gradient of operations involving the scalar, which is essential for tasks like gradient descent in machine learning models.
- **SIMD Operations**: Leveraged in operations involving scalars and vectors to boost computational efficiency, particularly in scenarios with large datasets.

## SIMD Operations

SIMD (Single Instruction, Multiple Data) operations are a key feature of this project, designed to enhance the performance of tensor computations. SIMD allows for the parallel processing of multiple data points with a single instruction, making tensor operations faster and more efficient, especially when dealing with large tensors.

### Benefits of SIMD Operations

- **Parallelism**: SIMD enables multiple tensor elements to be processed simultaneously, reducing computation time.
- **Optimized Performance**: Particularly effective in element-wise operations and large-scale tensor manipulations.
- **Scalability**: The SIMD approach scales well with the size of the tensor, making it suitable for high-performance computing tasks.

## Example Usage

### Tensor Operations

```java
public class Main {
    public static void main(String[] args) {
        Float[][] matrix1 = {{1, 2}, {3, 4}};
        
        ITensor tensor1 = new TensorVector(matrix1);
        ITensor tensor2 = new TensorScalar(3.4);
        
        // Element-wise addition
        ITensor resultAdd = tensor1.add(tensor2);
        System.out.println("Addition Result: " + resultAdd.getData());
        
        // Element-wise subtraction
        ITensor resultSubtract = tensor1.subtract(tensor2);
        System.out.println("Subtraction Result: " + resultSubtract.getData());
        
        // Element-wise multiplication
        ITensor resultMultiply = tensor1.multiply(tensor2);
        System.out.println("Multiplication Result: " + resultMultiply.getData());
        
        // Element-wise division
        ITensor resultDivide = tensor1.divide(tensor2);
        System.out.println("Division Result: " + resultDivide.getData());
    }
}
```

### Slicing

```java
public class Main {
    public static void main(String[] args) {
        Integer[][] matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        TensorVector tensor = new TensorVector(matrix);
        
        // Slice the tensor
        ITensor slice = tensor.slice(new JarvisPairs(0, 2), new JarvisPairs(1, 3));
        System.out.println("Sliced Tensor: " + slice.getData());
    }
}
```

### Gradient Calculation

```java
public class Main {
    public static void main(String[] args) {
        TensorScalar scalar1 = new TensorScalar(2);
        TensorScalar scalar2 = new TensorScalar(3);
        
        // Compute scalar addition and its gradient
        TensorScalar result = (TensorScalar) scalar1.add(scalar2);
        result.backPropagate();
        
        // Print the gradients
        System.out.println("Gradient of Scalar1: " + scalar1.getGradient().getData());
        System.out.println("Gradient of Scalar2: " + scalar2.getGradient().getData());
    }
}
```