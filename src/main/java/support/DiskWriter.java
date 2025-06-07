package support;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class DiskWriter {
    private final DataOutputStream       out;
    private final BlockingQueue<long[]>  queue = new LinkedBlockingQueue<>();
    private final Thread                 writer;

    public DiskWriter(Path outFile) throws IOException {
        this.out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(outFile))
        );
        this.writer = new Thread(() -> {
            try {
                while (true) {
                    long[] batch = queue.take();
                    if (batch.length == 0) break;
                    for (long b : batch) {
                        out.writeLong(b);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try { out.close(); } catch (IOException ignored) {}
            }
        }, "DiskWriter");
        writer.start();
    }

    public void push(long[] terminals) {
        queue.add(terminals);
    }

    public synchronized void write(long board, byte score) throws IOException {
        out.writeLong(board);
        out.writeByte(score);
    }

    public void finish() throws InterruptedException {
        queue.add(new long[0]);
        writer.join();
    }
}
