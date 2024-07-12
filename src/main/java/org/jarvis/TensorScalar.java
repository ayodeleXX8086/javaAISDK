package org.jarvis;

import org.jarvis.exceptions.JarvisRuntimeException;

import java.util.*;

public class TensorScalar implements ITensor {

    private final Number value;
    private Double gradient = 0d;

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
            this.value =((Number) value);
        } else if (value instanceof TensorScalar) {
            this.value = ((TensorScalar) value).value;
        } else {
            throw new JarvisRuntimeException("Cannot initialize " + value.getClass().getName() + " in TensorScalar ");
        }
    }

    @Override
    public ITensor getGradient() {
        return new TensorScalar(gradient);
    }

    public Double getValue() {
        return  (this.value).doubleValue();
    }

    @Override
    public ITensor add(ITensor iTensor) {
        if (iTensor instanceof TensorScalar tensorScalarleft) {
            var result = new TensorScalar((Double) tensorScalarleft.getValue() + this.getValue(), this, tensorScalarleft);
            result.backPropagateRun = () -> {
                tensorScalarleft.gradient += (1.0 * result.gradient);
                this.gradient += (1.0 * result.gradient);
            };
            return result;
        } else if (iTensor instanceof TensorVector tensorVector) {
            return tensorVector.add(this);
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
    public ITensor subtract(ITensor iTensor) {
        if (iTensor instanceof TensorScalar tensorScalarleft) {
            var result = new TensorScalar(this.getValue() - tensorScalarleft.getValue(), this, tensorScalarleft);
            result.backPropagateRun = () -> {
                tensorScalarleft.gradient -= (1.0 * result.gradient);
                this.gradient += (1.0 * result.gradient);
            };
            return result;
        } else if (iTensor instanceof TensorVector tensorVector) {
            var negativeOne = new TensorScalar(-1.0);
            var leftOperand = tensorVector.multiply(negativeOne);
            return leftOperand.add(this);
        }
        throw new JarvisRuntimeException("ITensor instance subtract operator is not implemented yet.");
    }

    @Override
    public ITensor multiply(ITensor iTensor) {
        if ((iTensor instanceof TensorScalar tensorScalarleft)) {
            var result = new TensorScalar(this.getValue() * tensorScalarleft.getValue(), this, tensorScalarleft);
            result.backPropagateRun = () -> {
                tensorScalarleft.gradient += this.getValue() * result.gradient;
                this.gradient += tensorScalarleft.getValue() * result.gradient;
            };
            return result;
        } else if (iTensor instanceof TensorVector tensorVector) {
            return tensorVector.multiply(this);
        }

        throw new JarvisRuntimeException("ITensor instance multiply operator is not implemented yet.");
    }

    @Override
    public ITensor pow(Number exp) {
        Double doubleExp = exp.doubleValue();
        var baseValue = this.getValue();
        var resultValue = Math.pow(baseValue, doubleExp);
        var result = new TensorScalar(resultValue, this, null);
        result.backPropagateRun = () -> {
            this.gradient += (doubleExp * Math.pow(baseValue, doubleExp) - 1) * result.gradient;
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
        this.gradient = 1d;
        topologicalSort(tensors, new HashSet<>(), this);
        Collections.reverse(tensors);
        for (var child : tensors) {
            child.backPropagateRun.apply();
        }
    }

    @Override
    public Object getData() {
        return (this.value).doubleValue();
    }
}
