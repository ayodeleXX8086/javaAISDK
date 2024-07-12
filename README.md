# TensorVector Library

The `TensorVector` library provides a way to handle high-dimensional matrices and compute gradients for these matrices. While it doesn't offer as many elaborate operations as other tensor libraries like NumPy, it provides essential functionalities for managing and manipulating multi-dimensional arrays.

## Features

- **High-Dimensional Matrix Handling**: Supports initialization and manipulation of multi-dimensional matrices.
- **Element-Wise Operations**: Supports addition, subtraction, multiplication, and division operations on tensors.
- **Gradient Calculation**: Enables automatic differentiation to compute gradients.
- **Broadcasting**: Implements broadcasting for element-wise operations on matrices of different shapes.

## Limitations

- **Limited Operations**: Doesn't support as many operations as more comprehensive tensor libraries like NumPy or TensorFlow.
- **Error Handling**: Limited error messages and checks.

## Installation

To use the `TensorVector` library, you need to include the following files in your project:

- `TensorVector.java`
- `TensorScalar.java`
- `ITensor.java` (interface)
- `JarvisRuntimeException.java` (exception class)
- `TupleRecord.java` (helper class)
- `JarvisPairs.java` (helper class)

## Usage

### Initialization

You can initialize a `TensorVector` with nested arrays or a `TensorScalar` with a numerical value.

```java
import org.jarvis.*;

public class Main {
    public static void main(String[] args) {
        // Initialize a 2D tensor
        Double[][] matrix = {{1, 2}, {3, 4}};
        TensorVector tensor = new TensorVector(matrix);
        
        // Initialize a scalar
        TensorScalar scalar = new TensorScalar(5);
        
        // Print tensor and scalar
        System.out.println("Tensor: " + tensor.getData());
        System.out.println("Scalar: " + scalar.getData());
    }
}
```

### Element-Wise Operations

You can perform element-wise addition, subtraction, multiplication, and division on tensors.

```java
import org.jarvis.ITensor;
import org.jarvis.TensorScalar;

public class Main {
    public static void main(String[] args) {
        Double[][] matrix1 = {{1, 2}, {3, 4}};

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

You can slice tensors to extract sub-tensors.

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

You can compute gradients using automatic differentiation.

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

