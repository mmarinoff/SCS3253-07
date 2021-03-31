package scs3253034.traffic.camera;

import org.springframework.util.FileCopyUtils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;

public class CameraDownloadThread implements Runnable {

    private final TrafficCamera camera;
    private final File outputDirectory;
    private final String timeId;
    private final CountDownLatch latch;

    public CameraDownloadThread(TrafficCamera camera, String timeId, File outputDirectory, CountDownLatch latch) {
        super();
        this.camera = camera;
        this.timeId = timeId;
        this.outputDirectory = outputDirectory;
        this.latch = latch;
    }

    public void run() {

        for (int i = 0; i < 10; i++) {
            try {
                download();
                break;
            } catch (Exception e) {
                if (i >= 9) {
                    System.err.println("XXX Failed download [" + e.getMessage() + "] going to retry [" + i + "]");
                }
            } finally {
                latch.countDown();
            }
        }
    }

    private void download() throws Exception {
        URL url = new URL(camera.getUrl());
        InputStream in = new BufferedInputStream(url.openStream());
        File file = new File(outputDirectory, sanitizeFileName(camera.getId()) + "-" + timeId + ".jpg");
        FileCopyUtils.copy(in, new FileOutputStream(file));
        in.close();
        Thread.sleep(7500);
    }

    private static String sanitizeFileName(String name) {
        return name
                .chars()
                .mapToObj(i -> (char) i)
                .map(c -> Character.isWhitespace(c) ? '_' : c)
                .filter(c -> Character.isLetterOrDigit(c) || c == '-' || c == '_')
                .map(String::valueOf)
                .collect(Collectors.joining());
    }
}
