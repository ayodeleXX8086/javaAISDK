package org.jarvis;

import org.jarvis.exceptions.JarvisRuntimeException;

import java.util.*;

public class TensorScalar implements ITensor {

    private final Number value;
    private Float gradient = 0f;

    private BackPropagate backPropagateRun = () -> {
    };
    private List<TensorScalar> tensorList;

    private final UUID uuid = UUID.randomUUID();

    private TensorScalar(Object value, TensorScalar tensorScalar1, TensorScalar tensorScalar2) {
        this(value);
        tensorList.add(tensorScalar1);
        tensorList.add(tensorScalar2);
    }

    public TensorScalar(Object value) {
        tensorList = new ArrayList<>();
        if (value instanceof Number) {
            this.value = ((Number) value);
        } else if (value instanceof TensorScalar) {
            this.value = ((TensorScalar) value).value;
        } else {
            throw new JarvisRuntimeException("Cannot initialize " + value.getClass().getName() + " in TensorScalar ");
        }
    }

    @Override
    public Object getGradient() {
        return gradient;
    }

    public Float getValue() {
        return (this.value).floatValue();
    }

    @Override
    public ITensor add(ITensor iTensor) {
        if (iTensor instanceof TensorScalar tensorScalarleft) {
            var result = new TensorScalar((Float) tensorScalarleft.getValue() + this.getValue(), this, tensorScalarleft);
            result.backPropagateRun = () -> {
                tensorScalarleft.gradient += (1 * result.gradient);
                this.gradient += (1 * result.gradient);
            };
            return result;
        } else if (iTensor instanceof TensorVector tensorVector) {
            return new TensorVector(new float[]{this.value.floatValue()}).add(tensorVector);
        }

        throw new JarvisRuntimeException("ITensor instance add operator is not implemented yet.");
    }

    @Override
    public String toString() {
        return "TensorScalar{" +
                "value=" + value +
                ", gradient=" + gradient +
                '}';
    }

    @Override
    public ITensor neg() {
        return this.multiply(new TensorScalar(-1));
    }

    @Override
    public ITensor subtract(ITensor iTensor) {
        if (iTensor instanceof TensorScalar tensorScalarRight) {
            return this.add(tensorScalarRight.neg());
        } else if (iTensor instanceof TensorVector tensorVector) {
            return new TensorVector(new float[]{this.value.floatValue()}).subtract(tensorVector);
        }
        throw new JarvisRuntimeException("ITensor instance subtract operator is not implemented yet.");
    }

    @Override
    public ITensor multiply(ITensor iTensor) {
        if ((iTensor instanceof TensorScalar tensorScalarRight)) {
            var result = new TensorScalar(this.getValue() * tensorScalarRight.getValue(), this, tensorScalarRight);
            result.backPropagateRun = () -> {
                tensorScalarRight.gradient += this.getValue() * result.gradient;
                this.gradient += tensorScalarRight.getValue() * result.gradient;
            };
            return result;
        } else if (iTensor instanceof TensorVector tensorVector) {
            return new TensorVector(new float[]{this.value.floatValue()}).multiply(tensorVector);
        }

        throw new JarvisRuntimeException("ITensor instance multiply operator is not implemented yet.");
    }

    @Override
    public ITensor pow(Number exp) {
        Float doubleExp = exp.floatValue();
        var baseValue = this.getValue();
        var resultValue = Math.pow(baseValue, doubleExp);
        var result = new TensorScalar(resultValue, this, null);
        result.backPropagateRun = () -> {
            this.gradient += (doubleExp * (float) Math.pow(baseValue, doubleExp - 1)) * result.gradient;
        };
        return result;
    }


    @Override
    public ITensor divide(ITensor iTensor) {
        return this.multiply(iTensor.pow(-1));
    }

    @Override
    public String tensorID() {
        return uuid.toString();
    }

    private void topologicalSort(List<TensorScalar> tensorList, Set<String> seen, TensorScalar node) {
        if (node == null || seen.contains(node.tensorID())) {
            return;
        }
        seen.add(node.tensorID());
        for (TensorScalar tensor : node.tensorList) {
            topologicalSort(tensorList, seen, tensor);
        }
        tensorList.add(node);
    }

    @Override
    public void backPropagate() {
        List<TensorScalar> tensors = new ArrayList<>();
        this.gradient = 1f;
        topologicalSort(tensors, new HashSet<>(), this);
        Collections.reverse(tensors);
        for (var child : tensors) {
            child.backPropagateRun.apply();
        }
    }

    public void clearGradient(){
        List<TensorScalar> tensors = new ArrayList<>();
        this.gradient = 0f;
        topologicalSort(tensors, new HashSet<>(), this);
        Collections.reverse(tensors);
        for (var child : tensors) {
            child.gradient = 0f;
        }
    }

    @Override
    public Object getData() {
        return (this.value).floatValue();
    }

    public String toStringGradient() {
        return this.gradient.toString();
    }
}
