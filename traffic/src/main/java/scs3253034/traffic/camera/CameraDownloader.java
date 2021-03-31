package scs3253034.traffic.camera;

import java.io.File;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraDownloader {

    private final List<TrafficCamera> cameras;
    private final File outputDirectory;
    private final ExecutorService executorService = Executors.newWorkStealingPool();
    private final CountDownLatch latch;

    public CameraDownloader(ApplicationConfiguration configuration, CountDownLatch latch) {
        super();
        this.cameras = configuration.getCameras();
        this.outputDirectory = configuration.getOutputDirectory();
        this.latch = latch;
    }

    public void run() {
        int size = cameras.size();

        for (int i = 0; i < size; i++) {
            CameraDownloadThread thread = new CameraDownloadThread(cameras.get(i), System.currentTimeMillis()
                    + "", outputDirectory, latch);
            executorService.submit(thread);
        }
    }
}
