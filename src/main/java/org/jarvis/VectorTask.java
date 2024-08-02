package org.jarvis;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.concurrent.RecursiveAction;
import java.util.function.BiFunction;

public class VectorTask extends RecursiveAction {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;


    private final float[] a;
    private final float[] b;
    private final BiFunction<FloatVector, FloatVector, FloatVector> operationVector;

    private final BiFunction<Float, Float, Float> operationScalar;
    private final float[] result;
    private final int start;
    private final int end;
    private final int threshold;

    public VectorTask(float[] a, float[] b, BiFunction<FloatVector, FloatVector, FloatVector> operationVector, BiFunction<Float, Float, Float> operationScalar, float[] result, int start, int end, int threshold) {
        this.a = a;
        this.b = b;
        this.operationVector = operationVector;
        this.operationScalar = operationScalar;
        this.result = result;
        this.start = start;
        this.end = end;
        this.threshold = threshold;
    }

    @Override
    protected void compute() {
        if (end - start <= threshold) {
            computeDirectly();
        } else {
            int mid = (start + end) / 2;
            invokeAll(
                    new VectorTask(a, b, operationVector, operationScalar, this.result, start, mid, threshold),
                    new VectorTask(a, b, operationVector, operationScalar, this.result, mid, end, threshold)
            );
        }
    }

    private void computeDirectly() {
        int i = start;
        int upperBound = SPECIES.loopBound(end - start);
        for (; i < upperBound; i += SPECIES.length()) {
            var v1 = FloatVector.fromArray(SPECIES, a, i);
            var v2 = FloatVector.fromArray(SPECIES, b, i);
            var result = operationVector.apply(v1, v2);
            result.intoArray(this.result, i);
        }
        for (; i < end; i++) {
            this.result[i] = operationScalar.apply(a[i], b[i]);
        }
    }
}
