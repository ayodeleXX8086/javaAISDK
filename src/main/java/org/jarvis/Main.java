package org.jarvis;


import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;

public class Main {
    static final VectorSpecies<Integer> SPECIES = IntVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float> PREFERRED_SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void main(String[] args) {

        // Define Tensor A
        Float[][][] tensorA = {
                {
                        {1.0F, 2.0f, 3.0f, 4.0f},
                        {5.0f, 6.0f, 7.0f, 8.0f},
                        {9.0f, 10.0f, 11.0f, 12.0f}
                },
                {
                        {13.0f, 14.0f, 15.0f, 16.0f},
                        {17.0f, 18.0f, 19.0f, 20.0f},
                        {21.0f, 22.0f, 23.0f, 24.0f}
                }
        };
//        var lst = getSizes(tensorA);
//        int size = lst.stream().reduce((a, b) -> a * b).get();
//        Object array1 = Array.newInstance(Object.class, size);
//        initializeMatrix(tensorA, array1, new AtomicInteger());
//        printArray(array1, 0);
//        System.out.println(size + " " + Array.getLength(array1));


        // Define Tensor B
        float[][] tensorB = {
                {1, 1, 1, 1},
                {2, 2, 2, 2},
                {3, 3, 3, 3}
        };

        float[] tensor3D = {1F, 2, 3, 4};
        float[][] tensor1D3D = {{1}, {2}, {3}};

        float[][][][][] arrayFiveD = {
                {
                        {
                                {
                                        {0, 1},
                                        {2, 3}
                                },
                                {
                                        {4, 5},
                                        {6, 7}
                                }
                        },
                        {
                                {
                                        {8, 9},
                                        {10, 11}
                                },
                                {
                                        {12, 13},
                                        {14, 15}
                                }
                        }
                },
                {
                        {
                                {
                                        {16, 17},
                                        {18, 19}
                                },
                                {
                                        {20, 21},
                                        {22, 23}
                                }
                        },
                        {
                                {
                                        {24, 25},
                                        {26, 27}
                                },
                                {
                                        {28, 29},
                                        {30, 31}
                                }
                        }
                }
        };


        // Create TensorVector instances
        TensorVector vectorA = new TensorVector(tensorA);
        System.out.println(vectorA);
        TensorVector fiveDvector = new TensorVector(arrayFiveD);
        System.out.println("Matrix object ID " + fiveDvector.getData());
        System.out.println(fiveDvector);
        System.out.println(fiveDvector);
        System.out.println(vectorA.get(1));
        System.out.println(vectorA.get(1, 1));
        System.out.println(fiveDvector.get(1, 1));
        TensorVector tensorVector1 = new TensorVector(tensorB);
        TensorVector tensorVector2 = new TensorVector(tensor3D);
        TensorVector tensorVector3 = new TensorVector(tensor1D3D);
        System.out.println(tensorVector1 + " " + tensorVector2);
        System.out.println(tensorVector1.multiply(tensorVector3));
        float[][] tensorA1 = {
                {1, 2, 3}
        };
        TensorVector tensorVector4 = new TensorVector(tensorA1);
        System.out.println(tensorVector3.multiply(tensorVector4));

        float[][][] array3D = {
                {{8, 9, 4}},
                {{4, 5, 6}}
        };
        TensorVector tensorVector5 = new TensorVector(array3D);
        float[][] array2D = {
                {1, 2, 3},
                {4, 5, 6}
        };
        TensorVector tensorVector6 = new TensorVector(array2D);
        System.out.println(tensorVector5);
        System.out.println(tensorVector6);
        System.out.println(tensorVector5.multiply(tensorVector6));

        float[][][] array6 = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
        TensorVector tensorVector7 = new TensorVector(array6);
        float[][] array6A = {{1, 2, 3}};
        TensorVector tensorVector8 = new TensorVector(array6A);

        System.out.println(tensorVector7.multiply(tensorVector8));
    }

    // Helper method to print the tensor

}
