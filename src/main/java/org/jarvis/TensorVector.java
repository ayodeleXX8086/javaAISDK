package org.jarvis;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.jarvis.exceptions.JarvisRuntimeException;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class TensorVector implements ITensor {

    private final Integer[] sizes;
    private final float[] matrix;

    private final Map<Object, Object> cache;
    private final String tensorId;
    private final Integer totalSize;

    private final StringBuilder optimizedToString;
    private final List<TensorVector> gradients;

    private BackPropagate backPropagateRun = () -> {
    };


    // Common constructor that all other constructors delegate to
    private TensorVector(float[] matrix, Integer[] sizes, List<TensorVector> gradients, boolean flattenSize) {
        this.sizes = sizes;
        this.totalSize = reduceMultiple(this.sizes);
        this.matrix = matrix;
        this.optimizedToString = new StringBuilder();
        this.tensorId = UUID.randomUUID().toString();
        this.cache = new HashMap<>();
        this.gradients = gradients;
    }

    // Constructor 1: Uses default sizes
    public TensorVector(Object matrix) {

        this(initializeMatrix(matrix, reduceMultiple(getSizes(matrix))), getSizes(matrix), new ArrayList<>(), true);
    }

    // Constructor 2: Uses provided sizes
    public TensorVector(float[] matrix, Integer[] sizes) {
        this(matrix, sizes, new ArrayList<>(), false);
    }

    // Constructor 3: Uses provided sizes and gradients
    public TensorVector(float[] matrix, Integer[] sizes, TensorVector... gradients) {
        this(matrix, sizes, Arrays.asList(gradients), false);
    }

    static Integer reduceMultiple(Integer[] sizes) {
        if (sizes.length == 0) return 0;
        return Arrays.stream(sizes).reduce((a, b) -> a * b).orElse(1);
    }

    private static List<Float> stretchBroadcast(List<Float> arr, int index, int prev, Integer[] originalSize, Integer[] targetSize) {
        if (index >= targetSize.length) return arr;
        var originalSizeCurr = index < originalSize.length ? originalSize[index] : 1;
        var targetSizeCurr = targetSize[index];
        if (originalSizeCurr != targetSizeCurr) {
            List<Float> curr = new ArrayList<>();
            for (int i = 0; i < arr.size(); i += prev) {
                for (int j = 0; j < targetSizeCurr; j++) {
                    curr.addAll(arr.subList(i, Math.min(i + prev, arr.size())));
                }
            }
            return stretchBroadcast(curr, index + 1, prev * targetSizeCurr, originalSize, targetSize);
        } else {
            List<Float> curr = new ArrayList<>(arr);
            return stretchBroadcast(curr, index + 1, prev * targetSizeCurr, originalSize, targetSize);
        }
    }

    public static <T> T[] invertArray(T[] array, Class<T> tClass) {
        T[] result = (T[]) Array.newInstance(tClass, array.length);
        for (int i = 0; i < array.length; i++) {
            Array.set(result, array.length - 1 - i, Array.get(array, i));
        }
        return result;
    }

    // Stretching method for broadcasting
    private static float[] stretch(float[] values, Integer[] originalSize, Integer[] targetSize) {
        List<Float> arr = new ArrayList<>();
        for (float value :
                values) {
            arr.add(value);
        }

        Integer[] invertOriginal = invertArray(originalSize, Integer.class);
        Integer[] invertTarget = invertArray(targetSize, Integer.class);
        var arr1 = stretchBroadcast(arr, 0, 1, invertOriginal, invertTarget);
        float[] result = new float[arr1.size()];
        AtomicInteger atomicInteger = new AtomicInteger();
        arr1.stream().forEach(e -> {
            result[atomicInteger.getAndIncrement()] = e;
        });
        return result;
    }

    private static float[] initializeMatrix(Object value, Integer size) {
        float[] matrix = (float[]) Array.newInstance(float.class, size);
        flattenMatrix(value, matrix, new AtomicInteger());
        return matrix;
    }

    // Method to flatten a matrix into a 1D array
    static void flattenMatrix(Object value, Object matrix, AtomicInteger index) {
        // Check if the input value is an array
        if (value.getClass().isArray()) {
            // Get the length of the array
            int length = Array.getLength(value);
            // Recursively flatten each element of the array
            for (int i = 0; i < length; i++) {
                flattenMatrix(Array.get(value, i), matrix, index);
            }
        } else if (value instanceof Number) {
            // Set the value in the matrix at the current index and increment the index
            Array.set(matrix, index.getAndIncrement(), ((Number) value).floatValue());
        } else if (value instanceof TensorScalar) {
            Array.set(matrix, index.getAndIncrement(), ((TensorScalar) value).getValue());
        } else {
            // Throw an exception if the value is of an unsupported type
            throw new JarvisRuntimeException("Cannot initialize " + value.getClass().getName() + " in TensorVector");
        }
    }

    //
    private static Integer[] getSizes(Object arr) {
        List<Integer> sizes = initializeSizeHelper(arr);
        if (sizes.isEmpty()) throw new JarvisRuntimeException("Cannot get the array size for a null");
        Integer[] result = new Integer[sizes.size()];
        sizes.toArray(result);
        return result;
    }

    static List<Integer> initializeSizeHelper(Object arr) {
        List<Integer> sizes = new ArrayList<>();
        if (arr.getClass().isArray()) {
            Integer size = null;
            for (int i = 0; i < Array.getLength(arr); i++) {
                if (Array.get(arr, i).getClass().isArray()) {
                    size = size == null ? Array.getLength(Array.get(arr, i)) : size;
                    if (size == Array.getLength(Array.get(arr, i))) {
                        if (sizes.isEmpty()) {
                            sizes = initializeSizeHelper(Array.get(arr, i));
                        }
                        if (!sizes.equals(initializeSizeHelper(Array.get(arr, i)))) {
                            throw new JarvisRuntimeException("Cannot initialize TensorVector expected a size of " + sizes.size());
                        }
                    } else {
                        throw new JarvisRuntimeException("Cannot initialize TensorVector expected a size of " + size);
                    }
                } else if (Array.get(arr, i) instanceof TensorVector tensorVector) {
                    if (size != null && !size.equals(tensorVector.sizes[0]))
                        throw new JarvisRuntimeException("Cannot initialize TensorVector expected a size of " + size);
                    if (sizes.size() != tensorVector.sizes.length - 1)
                        throw new JarvisRuntimeException("Cannot initialize TensorVector expected a size of " + sizes.size());
                    if (tensorVector.sizes.length > 1) {
                        for (int j = 0; j < sizes.size(); j++) {
                            if (!tensorVector.sizes[j + 1].equals(sizes.get(j)))
                                throw new JarvisRuntimeException("Cannot initialize TensorVector expected a size of " + sizes.get(j));
                        }
                    }
                }
            }
            sizes.add(0, Array.getLength(arr));
            return sizes;
        } else if (arr instanceof TensorVector tensorVector) {
            return List.of(tensorVector.sizes);
        }
        throw new JarvisRuntimeException("Expected type should be TensorVector or Array");
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

    private boolean isValidBroadcast(Integer[] shape1, Integer[] shape2) {
        int len1 = shape1.length;
        int len2 = shape2.length;
        int maxLength = Math.max(len1, len2);
        for (int i = 0; i < maxLength; i++) {
            int dim1 = i < len1 ? shape1[len1 - 1 - i] : 1;
            int dim2 = i < len2 ? shape2[len2 - 1 - i] : 1;
            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                return false;
            }
        }
        return true;
    }

    private int getOffset(int[] indices) {
        int offset = 0;
        int multiplier = 1;
        for (int i = this.sizes.length - 1; i >= 0; i--) {
            if (i < indices.length) {
                offset += indices[i] * multiplier;
            }
            multiplier *= this.sizes[i];
        }
        return offset;
    }

    public Integer[] broadcastShape(Integer[] shape1, Integer[] shape2) {
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
        if (indices.length == 0) {
            throw new IllegalArgumentException("At least one index must be provided.");
        }

        int offset = getOffset(indices);
        if (indices.length == this.sizes.length) {
            // Single element retrieval
            return new TensorScalar(Array.get(this.matrix, offset));
        } else if (indices.length < this.sizes.length) {
            // Sub-array retrieval
            Integer[] subSizes = Arrays.copyOfRange(this.sizes, indices.length, this.sizes.length);
            int subArraySize = Arrays.stream(subSizes).reduce((a, b) -> a * b).orElse(1);
            float[] subArray = (float[]) Array.newInstance(float.class, subArraySize);
            for (int i = 0; i < subArraySize; i++) {
                Array.set(subArray, i, Array.get(this.matrix, offset + i));
            }
            return new TensorVector(subArray, subSizes);
        } else {
            throw new IllegalArgumentException("Too many indices provided.");
        }
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
        if (indices.length > sizes.length) {
            throw new IllegalArgumentException("Incorrect number of indices.");
        }
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= sizes[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for dimension " + i);
            }
        }
    }


    @Override
    public ITensor getGradient() {
        return null;
    }

    public Integer[] getSizes() {
        return sizes;
    }

    private ITensor elementWiseOperation(Object a, Object b, BiFunction<FloatVector, FloatVector, FloatVector> operationVector, BiFunction<Float, Float, Float> operationScalar, TensorOperation operation) {
        if ((a instanceof TensorVector) && (b instanceof TensorVector)) {
            TensorVector tv1 = (TensorVector) a;
            TensorVector tv2 = (TensorVector) b;

            if (!isValidBroadcast(tv1.sizes, tv2.sizes)) {
                throw new JarvisRuntimeException("Shapes are not broadcastable");
            }

            Integer[] resultShape = broadcastShape(tv1.sizes, tv2.sizes);
            float[] stretchedA = stretch(tv1.matrix, tv1.sizes, resultShape);
            float[] stretchedB = stretch(tv2.matrix, tv2.sizes, resultShape);

            float[] result = new float[reduceMultiple(resultShape)];
            int threshold = 10_000;
            try (ForkJoinPool pool = new ForkJoinPool()) {
                pool.invoke(new VectorTask(stretchedA, stretchedB, operationVector, operationScalar, result, 0, result.length, threshold));
            }

            return new TensorVector(result, resultShape);
        }
        throw new JarvisRuntimeException("Operation not supported " + a.getClass().getName() + " cannot perform " + operation.getOperationName() + " operation on " + b.getClass().getName());
    }

    @Override
    public ITensor add(ITensor iTensor) {
        return this.executeOperation(iTensor, FloatVector::add, (o1, o2) -> o1 + o2, TensorOperation.Addition);
    }

    @Override
    public ITensor subtract(ITensor iTensor) {
        return this.executeOperation(iTensor, FloatVector::sub, (o1, o2) -> o1 - o2, TensorOperation.Subtraction);
    }

    @Override
    public ITensor divide(ITensor iTensor) {
        return this.executeOperation(iTensor, FloatVector::div, (o1, o2) -> o1 / o2, TensorOperation.Division);
    }

    @Override
    public ITensor multiply(ITensor iTensor) {
        return this.executeOperation(iTensor, FloatVector::mul, (o1, o2) -> o1 * o2, TensorOperation.Multiplication);
    }

    @Override
    public ITensor pow(Number exp) {
        return executeOperation(this, exp);
    }

    ITensor executeOperation(Object operand, Number exp) {
        if (operand instanceof TensorVector || operand.getClass().isArray()) {
            Object arr = operand.getClass().isArray() ? operand : ((TensorVector) operand).matrix;
            int len = Array.getLength(arr);
            Object result = Array.newInstance(Object.class, len);
            for (int i = 0; i < len; i++) {
                var selectedElement = Array.get(arr, i);
                Array.set(result, i, executeOperation(selectedElement, exp));
            }
            return new TensorVector(result);
        } else if (operand instanceof TensorScalar tensorScalar) {
            return tensorScalar.pow(exp);
        }
        throw new JarvisRuntimeException("Operation cannot be performed on " + operand.getClass().getName());
    }

    ITensor executeOperation(Object operand, BiFunction<FloatVector, FloatVector, FloatVector> operator, BiFunction<Float, Float, Float> operationScalar, TensorOperation operation) {
        if (operand instanceof TensorVector other) {
            if (!isValidBroadcast(this.sizes, other.sizes)) {
                throw new IllegalArgumentException("Shapes are not broadcast " + operation.getOperationName());
            }
            var result = elementWiseOperation(this, other, operator, operationScalar, operation);
            return result;
        } else if (operand instanceof TensorScalar) {
            var result = elementWiseOperation(this.matrix, (ITensor) operand, operator, operationScalar, operation);
            return result;
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


    private String buildMatrixRepresentation(int idx, AtomicInteger atomicInteger) {
        StringBuilder stringBuilder = new StringBuilder();
        var breakLine = "\n";
        var tabs = "\t".repeat(idx);
        stringBuilder.append(breakLine);
        stringBuilder.append(tabs);
        stringBuilder.append("[");
        if (idx == this.sizes.length - 1) {
            List<String> lst = new ArrayList<>();
            for (int i = 0; i < this.sizes[idx]; i++) {
                lst.add(String.format(" %.1f", (float) Array.get(this.matrix, atomicInteger.getAndIncrement())));
            }
            stringBuilder.append(String.join(", ", lst).strip());
            stringBuilder.append("]");
        } else {
            for (int i = 0; i < this.sizes[idx]; i++) {
                stringBuilder.append(buildMatrixRepresentation(idx + 1, atomicInteger));
                if (i != this.sizes[idx] - 1) {
                    stringBuilder.append(",");
                }
            }
            stringBuilder.append("\n").append("\t".repeat(idx)).append("]");
        }
        return stringBuilder.toString();
    }

    private String buildMatrixRepresentationOptimized() {
        if (this.optimizedToString.isEmpty()) {
            this.optimizedToString.append(buildMatrixRepresentation(0, new AtomicInteger()));
        }
        return this.optimizedToString.toString();
    }

    @Override
    public String toString() {
        return "TensorVector{" +
                "matrix = " + buildMatrixRepresentationOptimized() +
                ", tensorId='" + tensorId + '\'' +
                ", shape=" + Arrays.toString(this.sizes) +
                '}';
    }
}
