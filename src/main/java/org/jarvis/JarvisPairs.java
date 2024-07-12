package org.jarvis;

public record JarvisPairs(int start, int end) {
    public JarvisPairs(int start) {
        this(start, Integer.MAX_VALUE);
    }
}
