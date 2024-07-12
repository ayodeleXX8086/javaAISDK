package org.jarvis;

public enum TensorOperation {
    Multiplication("Multiplication"),

    Addition("Addition"),

    Division("Division"),

    Subtraction("Subtraction"),

    Power("Power");

    private final String operationName;

    TensorOperation(String operationName) {
        this.operationName = operationName;
    }

    public String getOperationName() {
        return operationName;
    }
}
