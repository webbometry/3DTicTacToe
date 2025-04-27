package precomputing.minimax.canonicalization;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class Canonicalization {
    private final DuplicateFilter dupFilter = new DuplicateFilter();
    private final PermutationCanonicalizer permCanon = new PermutationCanonicalizer();
    private final RotationCanonicalizer rotCanon;

    public Canonicalization(Path rotationMapsPath) throws IOException {
        this.rotCanon = new RotationCanonicalizer(rotationMapsPath);
    }

    /**
     * Runs all three steps—duplicate‐removal, permutation‐collapse, rotation‐collapse—in parallel.
     */
    public List<String> canonicalize(List<String> boards) {
        // 1) drop boards with repeated chars
        List<String> noDups = dupFilter.filter(boards);

        // 2) remove pure permutations
        List<String> noPerms = permCanon.filter(noDups);

        // 3) remove rotationally equivalent
        return rotCanon.canonicalize(noPerms);
    }
}