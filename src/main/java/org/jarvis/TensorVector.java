package org.jarvis;

import org.jarvis.exceptions.JarvisRuntimeException;

import java.lang.reflect.Array;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

public class TensorVector implements ITensor {

    private final Integer[] sizes;
    private final Object matrix;

    private final Map<String, ITensor> cache;
    private final String tensorId;

    public TensorVector(Object matrix) {
        var result = initializeMatrix(matrix);
        if (result.getData1().equals(0)) throw new JarvisRuntimeException("Matrix size is cannot be zero");
        this.matrix = result.getData2();
        this.sizes = initializeSize(matrix);
        this.tensorId = UUID.randomUUID().toString();
        this.cache = new HashMap<String, ITensor>();
    }

    private TupleRecord<Integer, Object> initializeMatrix(Object value) {
        if (value.getClass().isArray()) {
            int length = Array.getLength(value);
            Object newArray = Array.newInstance(Object.class, length);
            Integer size = null;
            for (int i = 0; i < length; i++) {
                var result = initializeMatrix(Array.get(value, i));
                if (size != null && !size.equals(result.getData1()))
                    throw new JarvisRuntimeException("The length tensor doesn't match");
                Array.set(newArray, i, result.getData2());
                size = result.getData1();
            }
            return new TupleRecord<>(length, newArray);
        } else if (value instanceof Number) {
            return new TupleRecord<>(0, new TensorScalar(value));
        } else if (value instanceof TensorScalar) {
            return new TupleRecord<>(0, value);
        } else if (value instanceof TensorVector) {
            return initializeMatrix(((TensorVector) value).matrix);
        } else {
            throw new JarvisRuntimeException("Cannot initialize " + value.getClass().getName() + " in TensorVector");
        }
    }

    //
    private Integer[] initializeSize(Object arr) {
        List<Integer> sizes = new ArrayList<>();
        while (arr != null && arr.getClass().isArray()) {
            int size = Array.getLength(arr);
            sizes.add(size);
            arr = size > 0 ? Array.get(arr, 0) : null;
        }
        if (sizes.isEmpty()) throw new JarvisRuntimeException("Cannot get the array size for a null");
        Integer[] result = new Integer[sizes.size()];
        sizes.toArray(result);
        return result;
    }

    private boolean isMatrixForBroadcast(Integer[] shapes1, Integer[] shapes2) {
        if (shapes1.length != shapes2.length || shapes1.length < 2) return false;
        for (int i = 0; i < shapes1.length; i++) {
            int dim1 = shapes1[i];
            int dim2 = shapes2[i];
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false;
            }
        }
        return true;
    }

    private static void copyArray(Object src, Object dest, int depth, JarvisPairs... dimensions) {
        if (depth > dimensions.length - 1) {
            return;
        } else if (depth == dimensions.length - 1) {
            if (!src.getClass().isArray() || !dest.getClass().isArray())
                throw new JarvisRuntimeException("Cannot copy because the vector dimension doesn't match");
            int start = dimensions[depth].start();
            int end = Math.min(Array.getLength(src), dimensions[depth].end());
            System.arraycopy(src, start, dest, start, end);
        } else {
            int start = dimensions[depth].start();
            int end = Math.min(dimensions[depth].end(), Array.getLength(src));
            for (int i = start; i < end; i++) {
                Object srcElem = Array.get(src, i);
                Object destElem = Array.get(dest, i);
                copyArray(srcElem, destElem, depth + 1, dimensions);
            }
        }
    }

    private List<Integer[]> createPairs(int[] arr) {
        List<Integer[]> pairs = new ArrayList<>();
        for (int i = 0; i < arr.length; i += 2) {
            if (i + 2 < arr.length) {
                pairs.add(new Integer[]{arr[i], arr[i + 1]});
            } else {
                pairs.add(new Integer[]{arr[i]});
            }
        }
        return pairs;
    }


    private Integer[] broadcastShape(Integer[] shape1, Integer[] shape2) {
        int len1 = shape1.length;
        int len2 = shape2.length;
        int maxLength = Math.max(len1, len2);
        Integer[] result = new Integer[maxLength];

        for (int i = 0; i < maxLength; i++) {
            int dim1 = i < len1 ? shape1[len1 - 1 - i] : 1;
            int dim2 = i < len2 ? shape2[len2 - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw new IllegalArgumentException("Shapes are not broadcastable");
            }
            result[maxLength - 1 - i] = Math.max(dim1, dim2);
        }
        return result;
    }

    private List<Integer[]> createPairs(Integer[] arr) {
        return createPairs(Arrays.stream(arr).mapToInt(i -> i).toArray());
    }

    @SuppressWarnings("unchecked")
    private Object createArray(Integer[] sizes, int index) {
        int size = sizes[index];
        Object array = java.lang.reflect.Array.newInstance(Object.class, size);
        if (index < sizes.length - 1) {
            for (int i = 0; i < size; i++) {
                java.lang.reflect.Array.set(array, i, createArray(sizes, index + 1));
            }
        }
        return array;
    }

    @SuppressWarnings("unchecked")
    public <T> void set(T value, int... indices) {
        checkIndices(indices);
        Object array = matrix;
        for (int i = 0; i < indices.length - 1; i++) {
            array = java.lang.reflect.Array.get(array, indices[i]);
        }
        java.lang.reflect.Array.set(array, indices[indices.length - 1], value);
    }

    @SuppressWarnings("unchecked")
    public ITensor get(int... indices) {
        checkIndices(indices);
        Object array = matrix;
        for (int i = 0; i < indices.length - 1; i++) {
            array = java.lang.reflect.Array.get(array, indices[i]);
        }
        Object object = java.lang.reflect.Array.get(array, indices[indices.length - 1]);


        return object.getClass().isArray() ? new TensorVector(object) : new TensorScalar(object);
    }

    public ITensor slice(JarvisPairs... pairs) {
        if (pairs.length == 0) {
            Object dest = Array.newInstance(Object.class, Arrays.stream(this.sizes).mapToInt(i -> i).toArray());
            Arrays.stream(this.sizes).map(e -> new JarvisPairs(0, e)).toArray();
            copyArray(this.matrix, dest, 0, Arrays.stream(this.sizes).map(e -> new JarvisPairs(0, e)).toArray(JarvisPairs[]::new));
            return new TensorVector(dest);
        }
        int start = pairs[0].start();
        if (start >= this.sizes.length) {
            return new TensorVector(Array.newInstance(Object.class, 0));
        }
        int[] sizes = new int[pairs.length];
        Object tmpMatrix = matrix;
        for (int i = 0; i < pairs.length; i++) {
            sizes[i] = Math.min(pairs[i].end() - pairs[i].start(), Array.getLength(tmpMatrix) - pairs[i].start());
            tmpMatrix = Array.get(tmpMatrix, 0);
        }
        Object dest = Array.newInstance(Object.class, sizes);
        copyArray(matrix, dest, 0, pairs);
        return new TensorVector(dest);
    }

    private void checkIndices(int[] indices) {
        if (indices.length != sizes.length) {
            throw new IllegalArgumentException("Incorrect number of indices.");
        }
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= sizes[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for dimension " + i);
            }
        }
    }

    private Object computeGradient(Object value) {
        if (value.getClass().isArray()) {
            int length = Array.getLength(value);
            Object newArray = Array.newInstance(Object.class, length);
            for (int i = 0; i < length; i++) {
                Array.set(newArray, i, initializeMatrix(Array.get(value, i)));
            }
            return new TensorVector(newArray);
        } else if (value instanceof TensorScalar) {
            return new TensorScalar(((TensorScalar) value).getGradient());
        } else if (value instanceof TensorVector) {
            return initializeMatrix(((TensorVector) value).matrix);
        } else {
            throw new JarvisRuntimeException("Cannot initialize " + value.getClass().getName() + " in TensorVector");
        }
    }

    @Override
    public ITensor getGradient() {
        if (this.cache.get("GRADIENT") != null) {
            return this.cache.get("GRADIENT");
        }
        return null;
    }

    private Object elementWiseOperation(Object a, Object b, BiFunction<ITensor, ITensor, ITensor> op, TensorOperation operation) {
        if (!((a.getClass().isArray() || a instanceof ITensor) && (b.getClass().isArray() || b instanceof ITensor)))
            throw new JarvisRuntimeException("Operation not supported " + a.getClass().getName() + " cannot perform " + operation.getOperationName() + " operation on " + b.getClass().getName());
        if ((a.getClass().isArray() || a instanceof TensorVector) && (b.getClass().isArray() || b instanceof TensorVector)) {
            Object tensor1 = a.getClass().isArray() ? a : ((TensorVector) a).matrix;
            Object tensor2 = b.getClass().isArray() ? b : ((TensorVector) b).matrix;
            int len = Math.max(Array.getLength(tensor1), Array.getLength(tensor2));
            Object result = Array.newInstance(Object.class, len);
            for (int i = 0; i < len; i++) {
                Object aElem = i < Array.getLength(tensor1) ? Array.get(tensor1, i) : Array.get(tensor1, 0);
                Object bElem = i < Array.getLength(tensor2) ? Array.get(tensor2, i) : Array.get(tensor2, 0);
                Array.set(result, i, elementWiseOperation(aElem, bElem, op, operation));
            }
            return result;
        } else if ((a.getClass().isArray() || a instanceof TensorVector)) {
            Object tensor = a.getClass().isArray() ? a : ((TensorVector) a).matrix;
            int len = Array.getLength(tensor);
            Object result = Array.newInstance(Object.class, len);
            for (int i = 0; i < len; i++) {
                Object aElem = Array.get(tensor, i);
                Array.set(result, i, elementWiseOperation(aElem, b, op, operation));
            }
            return result;
        } else if ((b.getClass().isArray() || b instanceof TensorVector)) {
            Object tensor = b.getClass().isArray() ? b : ((TensorVector) b).matrix;
            int len = Array.getLength(tensor);
            Object result = Array.newInstance(Object.class, len);
            for (int i = 0; i < len; i++) {
                Object elem = Array.get(tensor, i);
                Array.set(result, i, elementWiseOperation(a, elem, op, operation));
            }
            return result;
        } else {
            return op.apply((ITensor) a, (ITensor) b);
        }
    }

    @Override
    public ITensor add(ITensor iTensor) {
        return this.executeOperation(iTensor, ITensor::add, TensorOperation.Addition);
    }

    @Override
    public ITensor subtract(ITensor iTensor) {
        return this.executeOperation(iTensor, ITensor::subtract, TensorOperation.Subtraction);
    }

    @Override
    public ITensor divide(ITensor iTensor) {
        return this.executeOperation(iTensor, ITensor::divide, TensorOperation.Division);
    }

    @Override
    public ITensor multiply(ITensor iTensor) {
        return this.executeOperation(iTensor, ITensor::multiply, TensorOperation.Multiplication);
    }

    @Override
    public ITensor pow(Number exp) {
        return executeOperation(this, exp);
    }

    ITensor executeOperation(Object operand, Number exp){
        if(operand instanceof TensorVector || operand.getClass().isArray()){
            Object arr = operand.getClass().isArray()? operand : ((TensorVector) operand).matrix;
            int len = Array.getLength(arr);
            Object result = Array.newInstance(Object.class, len);
            for(int i=0;i<len;i++){
                var selectedElement = Array.get(arr, i);
                Array.set(result, i, executeOperation(selectedElement, exp));
            }
            return new TensorVector(result);
        } else if (operand instanceof TensorScalar tensorScalar) {
            return tensorScalar.pow(exp);
        }
        throw new JarvisRuntimeException("Operation cannot be performed on "+operand.getClass().getName());
    }
    ITensor executeOperation(Object operand, BiFunction<ITensor, ITensor, ITensor> operator, TensorOperation operation) {
        if (operand instanceof TensorVector other) {
            if (!isMatrixForBroadcast(this.sizes, other.sizes)) {
                throw new IllegalArgumentException("Shapes are not broadcastable for addition");
            }
            var result = elementWiseOperation(this.matrix, other.matrix, operator, operation);
            return new TensorVector(result);
        } else if (operand instanceof TensorScalar) {
            var result = elementWiseOperation(this.matrix, (ITensor) operand, operator, operation);
            return new TensorVector(result);
        }
        throw new JarvisRuntimeException("Unsupported right operand");
    }


    @Override
    public String tensorID() {
        return this.tensorId;
    }

    private static void runBackPropagate(Object value) {
        if (value.getClass().isArray()) {
            for (int i = 0; i < Array.getLength(value); i++) {
                runBackPropagate(Array.get(value, i));
            }
        } else if (value instanceof TensorVector tensorVector) {
            runBackPropagate(tensorVector.matrix);
        } else if (value instanceof TensorScalar tensorScalar) {
            tensorScalar.backPropagate();
        }
    }

    @Override
    public void backPropagate() {
        TensorVector.runBackPropagate(matrix);
    }

    @Override
    public Object getData() {
        return this.matrix;
    }

}
