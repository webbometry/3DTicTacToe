package precomputing.minimax.kernels.canonicalization;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import org.jocl.CL;

/**
 * Runs three phases:
 * 1) GPU‐filter invalid boards
 * 2) GPU‐sort & dedupe exact boards
 * 3) GPU‐collapse rotations (keep first‐seen)
 * <p>
 * Tracks how many boards each phase removed.
 */
public class Canonicalization {
    private final FilterInvalidBoards filterInvalid;
    private final SortAndUniqueBoards sortAndUnique;
    private final SignatureKernel sigK;
    private final RotationKernel rotK;

    /**
     * Holds final boards plus per‐phase removal counts
     */
    public static record Result(
            List<String> boards,
            int removedInvalid,
            int removedDuplicates,
            int removedRotations
    ) {
    }

    public Canonicalization(Path rotationMapsPath) throws IOException {
        CL.setExceptionsEnabled(true);
        filterInvalid = new FilterInvalidBoards();
        sortAndUnique = new SortAndUniqueBoards();
        sigK = new SignatureKernel();
        rotK = new RotationKernel(rotationMapsPath);
    }

    /**
     * @param boards list of move‐strings (all length == step)
     * @param step   number of moves in each string
     * @return Result containing the canonical boards and how many each step removed
     */
    public Result canonicalize(List<String> boards, int step) {
        // Phase 1: remove invalid
        int before1 = boards.size();
        List<String> valid = filterInvalid.filter(boards);
        int removedInvalid = before1 - valid.size();

        // Phase 2: sort chars & unique exact duplicates → TO DISK
        int before2 = valid.size();
        Path uniqFile = null;
        List<String> uniqSorted;
        try {
            uniqFile = sortAndUnique.sortAndUniqueToFile(valid);
            uniqSorted = Files.readAllLines(uniqFile);
        } catch (IOException e) {
            throw new RuntimeException("Failed during sort+unique to file", e);
        } finally {
            if (uniqFile != null) {
                try {
                    Files.deleteIfExists(uniqFile);
                } catch (IOException ignored) {
                }
            }
        }
        int removedDuplicates = before2 - uniqSorted.size();

        // Phase 3: signature → rotation collapse
        long[] sigs = sigK.computeSignatures(uniqSorted, step);
        int before3 = sigs.length;

        long[] survivors = rotK.collapseRotation(sigs);
        int removedRotations = before3 - survivors.length;

        // === NEW: map survivors back to strings via primitive sort + binary-search ===
        int N2 = sigs.length;
        // 1) build index array
        Integer[] idx = new Integer[N2];
        for (int i = 0; i < N2; i++) {
            idx[i] = i;
        }
        // 2) parallel sort indices by their signature value
        Arrays.parallelSort(idx, Comparator.comparingLong(i -> sigs[i]));

        // 3) build a sorted copy of the signatures for binary search
        long[] sortedSigs = new long[N2];
        for (int j = 0; j < N2; j++) {
            sortedSigs[j] = sigs[idx[j]];
        }

        // 4) for each survivor code, find its position and pick the original board
        List<String> result = new ArrayList<>(survivors.length);
        for (long code : survivors) {
            int pos = Arrays.binarySearch(sortedSigs, code);
            if (pos < 0) {
                throw new IllegalStateException("Survivor code not found: " + code);
            }
            result.add(uniqSorted.get(idx[pos]));
        }

        return new Result(result, removedInvalid, removedDuplicates, removedRotations);
    }

    /**
     * Release all GPU kernels/buffers when done
     */
    public void release() {
        filterInvalid.release();
        sortAndUnique.release();
        rotK.release();
        // SignatureKernel has no persistent mem to release
    }
}
