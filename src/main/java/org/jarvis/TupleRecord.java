package org.jarvis;

public class TupleRecord<F1, F2> {
    private final F1 data1;
    private final F2 data2;

    public TupleRecord(F1 data1, F2 data2) {
        this.data1 = data1;
        this.data2 = data2;
    }

    public F1 getData1() {
        return data1;
    }

    public F2 getData2() {
        return data2;
    }
}
