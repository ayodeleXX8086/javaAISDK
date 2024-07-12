package org.jarvis;

public interface ITensor {

    public ITensor getGradient();

    public ITensor add(ITensor iTensor);

    public ITensor subtract(ITensor iTensor);

    public ITensor divide(ITensor iTensor);

    public ITensor multiply(ITensor iTensor);

    public ITensor pow(Number exp);

    public String tensorID();

    public void backPropagate();

    public Object getData();
}
