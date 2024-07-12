package org.jarvis;

//import jdk.incubator.vector.IntVector;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.reflect.Array;
import java.util.*;

import static jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

public class Main {
    static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float> PREFERRED_SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void main(String[] args) {
//        Integer[][][] arr = {
//                {
//                        {30, 67, 80, 36},
//                        {34, 25, 13, 67},
//                        {9, 59, 70, 38}
//                },
//                {
//                        {38, 20, 57, 77},
//                        {90, 85, 45, 93},
//                        {0, 57, 52, 55}
//                }
//        };
//        System.out.println(Arrays.deepToString(arr));
//        Object src = arr;
//        List<Integer> size = getSizes(arr);
//        Integer[] sizeArray = size.toArray(new Integer[0]);
//        Integer[] intArray = new Integer[sizeArray.length];
//        for (int i = 0; i < sizeArray.length; i++) {
//            intArray[i] = sizeArray[i];
//        }
//        System.out.println(Arrays.toString(intArray));
//
//        Object dest = Array.newInstance(Object.class, Arrays.stream(reverseArray(intArray)).mapToInt(i -> i).toArray());
//        System.out.println(getSizes(dest));
//        int[] arr32 = new int[1000];
//        for (int i = 0; i < 1000; i++) {
//            arr32[i] = (int) (Math.random() * 100_000_00);
//        }
//
//        int[] arr33 = new int[1000];
//        for (int i = 0; i < 1000; i++) {
//            arr33[i] = (int) (Math.random() * 100_000_00);
//        }
//
//        System.out.println(Arrays.toString(addTwoVectorsWIthHigherShape(arr32, arr33)));
//        System.out.println(Arrays.toString(addTwoVectorArrays(arr32, arr33)));
//        int xy = 45;
//        float xyFloat = 45f;
//        var isValidInt = Integer.class.isInstance(xyFloat);
//        System.out.println(isValidInt);
////        List<Integer[]> list = new ArrayList<>();
////        list.add(new Integer[]{0, 2});
////        list.add(new Integer[]{0, 1});
////        list.add(new Integer[]{0, 3});
////        copyArray(src, dest, list, 0);
////
////        setValueArray(dest, 3, 0, 0, 0);
////
////        System.out.println(Array.get(Array.get(Array.get(dest, 0), 0), 0));
////        System.out.println(Array.get(Array.get(Array.get(src, 0), 0), 0));
////        System.out.println(new JarvisPairs(5));
////        Integer[] arr5 = new Integer[]{4};
////        System.out.println(Arrays.toString(reverseArray(arr5)));
////        transpose(dest, src, sizeArray, reverseArray(sizeArray), 0, sizeArray.length, 0, 0);
//        Stack<Integer> stack = new Stack<>();
//        transposeRecursive(src, dest, sizeArray, stack, 0);
//        System.out.println(Arrays.toString(reverseArray(sizeArray)));
//        System.out.println(Array.get(Array.get(Array.get(dest, 0), 0), 1));
//        System.out.println(Array.get(Array.get(Array.get(src, 0), 0), 1));
//
////        for (; i <= length - SPECIES.length(); i += SPECIES.length()) {
////            DoubleVector va = DoubleVector.fromArray(SPECIES, (double[]) a, i);
////            DoubleVector vb = DoubleVector.fromArray(SPECIES, (double[]) b, i);
////            DoubleVector vr = isAdd ? va.add(vb) : va.sub(vb);
////            vr.intoArray((double[]) result, i);
//        }

        // Define Tensor A
        Double[][][] tensorA = {
                {
                        {1.0, 2.0, 3.0, 4.0},
                        {5.0, 6.0, 7.0, 8.0},
                        {9.0, 10.0, 11.0, 12.0}
                },
                {
                        {13.0, 14.0, 15.0, 16.0},
                        {17.0, 18.0, 19.0, 20.0},
                        {21.0, 22.0, 23.0, 24.0}
                }
        };

        ITensor tensor = new TensorVector(new Object[][]{{1.2}, {1.3}});

        // Define Tensor B
        Double[][][] tensorB = {
                {
                        {1.0, 1.0, 1.0, 1.0},
                        {2.0, 2.0, 2.0, 2.0},
                        {3.0, 3.0, 3.0, 3.0}
                }
        };

        // Create TensorVector instances
        TensorVector vectorA = new TensorVector(tensorA);
        ITensor result = vectorA.pow(3);
        result.backPropagate();
        System.out.println("result");
        System.out.println();
        printTensor(result);
        System.out.println();
//        System.out.println("Vector A");
//        printTensor(vectorA);
        System.out.println("result 1");
        ITensor result1 = result.add(new TensorScalar(4.5));
        printTensor(result1);
        ITensor result2=result1.divide(new TensorScalar(4));
        System.out.println("Result 2");
        result2.backPropagate();
        printTensor(result2);
        System.out.println();
        System.out.println("Vector A");
        printTensor(vectorA);
//        TensorVector vectorB = new TensorVector(tensorB);
////
//        ITensor result = vectorA.multiply(new TensorScalar(5.6));
//        ITensor iTensor = result.add(new TensorScalar(4.5));
//        iTensor.backPropagate();
//        System.out.println("Addition result: ");
//        printTensor(iTensor);
//        System.out.println();
//        System.out.println("result multiple: ");
//        printTensor(result);
//        System.out.println();
//        System.out.println("result vectorA");
//        printTensor(vectorA);
//        System.out.println();
//        printTensor(vectorA.slice(new JarvisPairs[]{new JarvisPairs(0, 1)}));

    }

    // Helper method to print the tensor
    private static void printTensor(ITensor tensor) {
        printArray(tensor.getData(), 0);
    }

    // Recursive method to print nested arrays
    private static void printArray(Object array, int depth) {
        if (array.getClass().isArray()) {
            System.out.print("[");
            boolean comma = false;
            for (int i = 0; i < Array.getLength(array); i++) {
                if (Array.get(array, i).getClass().isArray()) {
                    printArray(Array.get(array, i), depth + 1);
                } else {
                    System.out.print(Array.get(array, i) + " ");
                    comma = true;
                }
            }
            System.out.print("]");
            if (comma) System.out.print(",");
        }
    }

    static int[] addTwoVectorArrays(int[] arr1, int[] arr2) {
        var v1 = IntVector.fromArray(SPECIES, arr1, 0);
        var v2 = IntVector.fromArray(SPECIES, arr2, 0);
        var result = v1.add(v2);
        return result.toArray();
    }

    static int[] addTwoVectorsWIthHigherShape(int[] arr1, int[] arr2) {
        int[] finalResult = new int[arr1.length];
        for (int i = 0; i < arr1.length; i += SPECIES.length()) {
            var v1 = IntVector.fromArray(SPECIES, arr1, i);
            var v2 = IntVector.fromArray(SPECIES, arr2, i);
            var result = v1.add(v2);
            result.intoArray(finalResult, i);
        }
        return finalResult;
    }

    private static Object get(Object array, Integer[] indices) {
        for (int i = 0; i < indices.length - 1; i++) {
            array = Array.get(array, indices[i]);
        }
        return Array.get(array, indices[indices.length - 1]);
    }

    private static List<Integer> getSizes(Object src1) {
        List<Integer> size = new ArrayList<>();
        while (src1 != null && src1.getClass().isArray()) {
            size.add(Array.getLength(src1));
            src1 = Array.getLength(src1) > 0 ? Array.get(src1, 0) : null;
        }
        return size;
    }

    private static Object get(Object array, List<Integer> indices) {
        for (int i = 0; i < indices.size() - 1; i++) {
            array = Array.get(array, indices.get(i));
        }
        return Array.get(array, indices.get(indices.size() - 1));
    }

    private static Object get(Object array, List<Integer> indices, int start, int end, boolean reverse) {
        for (; start < end; ) {
            int i = reverse ? end-- : start++;
            array = Array.get(array, indices.get(i));
        }
        return Array.get(array, indices.get(start));
    }


    private static void transpose90DegreesRecursive(Object src, Object dest, Integer[] srcIndices, Integer[] destIndices, int size, int depth) {
        if (depth == srcIndices.length - 2) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    srcIndices[depth] = i;
                    srcIndices[depth + 1] = j;
                    destIndices[depth] = j;
                    destIndices[depth + 1] = size - i - 1;
                    Array.set(dest, destIndices[depth + 1], get(src, srcIndices));
                }
            }
            return;
        }

        for (int i = 0; i < size; i++) {
            srcIndices[depth] = i;
            destIndices[depth] = i;
            transpose90DegreesRecursive(Array.get(src, i), Array.get(dest, i), srcIndices, destIndices, size, depth + 1);
        }
    }

    private static void transposeRecursive(Object src, Object dest, Integer[] srcPos, Stack<Integer> indices, int depth) {
        if (indices.size() == srcPos.length) {
            Object arrSrc = get(src, indices.subList(0, indices.size() - 1));
            Integer lastIndex = indices.get(indices.size() - 1);
            Object destSrc = get(dest, indices, 1, indices.size() - 1, true);
            Integer lastIndex1 = indices.get(0);
            Array.set(destSrc, lastIndex1, Array.get(arrSrc, lastIndex));
            return;
        }
        for (int i = 0; i < srcPos[depth]; i++) {
            indices.add(i);
            transposeRecursive(src, dest, srcPos, indices, depth + 1);
            indices.pop();
        }
    }


    private static Integer[] reverseArray(Integer[] arr) {
        Integer[] reversedArray = new Integer[arr.length];
        for (int i = 0; i <= arr.length / 2; i++) {
            int back = arr.length - 1 - i;
            reversedArray[i] = arr[back];
            reversedArray[back] = arr[i];
        }
        return reversedArray;
    }

    private List<Integer[]> createPairs(Integer[] arr) {
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

    private static void copyArray(Object src, Object dest, List<Integer[]> dimensions, int depth) {
        if (depth == dimensions.size() - 1) {
            int start = dimensions.get(depth)[0];
            int end = dimensions.get(depth).length > 1 ? dimensions.get(depth)[1] : Array.getLength(src);
            System.arraycopy(src, start, dest, start, end);
        } else {
            int start = dimensions.get(depth)[0];
            int end = dimensions.get(depth)[1];
            for (int i = start; i < end; i++) {
                Object srcElem = Array.get(src, i);
                Object destElem = Array.get(dest, i);
                copyArray(srcElem, destElem, dimensions, depth + 1);
            }
        }
    }

    private static void setValueArray(Object dest, Object value, int... index) {
        int lastIndex = index[index.length - 1];
        for (int i = 0; i < index.length - 1; i++) {
            dest = Array.get(dest, index[0]);
        }
        Array.set(dest, lastIndex, value);
    }
}
