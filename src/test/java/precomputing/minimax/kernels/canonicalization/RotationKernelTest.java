// src/test/java/precomputing/minimax/kernels/canonicalization/RotationKernelThoroughTest.java
package precomputing.minimax.kernels.canonicalization;

import org.jocl.CL;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

class RotationKernelTest {
    private static RotationKernel rotK;

    // some useful single‐bit constants
    private static final long CORNER0 = 1L << 0;
    private static final long CORNER2 = 1L << 2;
    private static final long CENTER = 1L << 13;

    @BeforeAll
    static void setup() throws IOException {
        // initialize JOCL & our kernel
        CL.setExceptionsEnabled(true);
        rotK = new RotationKernel(Paths.get("src/main/data/rotationMaps.txt"));
    }

    @AfterAll
    static void teardown() {
        rotK.release();
    }

    @Test
    void testEmpty() {
        long[] in = {};
        long[] out = rotK.collapseRotation(in);
        assertEquals(0, out.length);
    }

    @Test
    void testSingleElement() {
        long x = 0x123456789ABCL;
        long[] out = rotK.collapseRotation(new long[]{x});
        assertArrayEquals(new long[]{x}, out);
    }

    @Test
    void testAllIdentical() {
        long[] in = {CORNER0, CORNER0, CORNER0};
        long[] out = rotK.collapseRotation(in);
        assertArrayEquals(new long[]{CORNER0}, out,
                "All identical inputs should collapse to a single representative");
    }

    @Test
    void testTwoEquivalentCorners_FirstWins() {
        // CORNER0 and CORNER2 are in the same orbit under rotation
        long[] in = {CORNER0, CORNER2};
        long[] out = rotK.collapseRotation(in);
        assertArrayEquals(new long[]{CORNER0}, out,
                "Between two equivalent corners, the first one should be kept");
    }

    @Test
    void testTwoEquivalentCorners_ReversedOrder() {
        long[] in = {CORNER2, CORNER0};
        long[] out = rotK.collapseRotation(in);
        assertArrayEquals(new long[]{CORNER2}, out,
                "Even when reversed, the first input should be kept for that class");
    }

    @Test
    void testTwoNonEquivalent() {
        // CORNER0 vs CENTER are not rotation‐equivalent
        long[] in = {CORNER0, CENTER};
        long[] out = rotK.collapseRotation(in);
        assertArrayEquals(new long[]{CORNER0, CENTER}, out,
                "Non‐equivalent codes should both be preserved, in order");
    }

    @Test
    void testMixedClasses_OrderPreserved() {
        // classes: {CORNER0,CORNER2} and {CENTER}
        long[] in = {CENTER, CORNER0, CORNER2, CENTER};
        long[] out = rotK.collapseRotation(in);
        // first‐seen for CENTER is CENTER (at index 0),
        // first‐seen for corners is CORNER0 (at index 1)
        assertArrayEquals(new long[]{CENTER, CORNER0}, out,
                "Should yield one per class, in the order their first member appeared");
    }

    @Test
    void testCompositeTwoStonePattern() {
        // shape A: stones at {0,1}
        long shapeA = (1L << 0) | (1L << 1);
        // shape B = rotate(shapeA) for which 1→3 under one rotation:
        long shapeB = (1L << 0) | (1L << 3);

        // they are equivalent under one of the 24 rotations
        long[] in = {shapeA, shapeB};
        long[] out = rotK.collapseRotation(in);
        assertArrayEquals(new long[]{shapeA}, out,
                "Composite two‐stone patterns should also collapse, keeping the first");
    }

    @Test
    void testCompositeWithCenterMixed() {
        long shapeA = (1L << 0) | (1L << 1);
        long shapeB = (1L << 0) | (1L << 3);
        long[] in = {shapeA, CENTER, shapeB, CENTER, CORNER2};
        long[] out = rotK.collapseRotation(in);
        // classes: {shapeA,shapeB}, {CENTER}, {CORNER2}
        // first‐seen are shapeA, CENTER, CORNER2
        assertArrayEquals(new long[]{shapeA, CENTER, CORNER2}, out,
                "Multiple classes of composite and single‐bit codes should collapse correctly");
    }
}
