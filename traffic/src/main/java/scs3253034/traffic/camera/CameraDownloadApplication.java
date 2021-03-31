package scs3253034.traffic.camera;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;

public class CameraDownloadApplication {

    private ApplicationConfiguration configuration;

    public CameraDownloadApplication(ApplicationConfiguration configuration) {
        super();
        this.configuration = configuration;
    }

    public void run() {

    }

    public static List<TrafficCamera> load() throws Exception {
        return CameraListDownloader.download();
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("CameraDownloadApplication dir schedule");
            System.err.println("-dir output directory for the files to download");
            System.err.println("-schedule how often the job should run (time in ms). 5 minutes would be = 5 * 60 * 1000");
            System.exit(-1);
        }

        File dir = new File(args[0]);

        if (!dir.exists()) {
            System.err.println("Directory [" + args[0] + "] does not exist. Create and try again.");
            System.exit(-1);
        }

        long time = 0;

        try {
            time = Long.parseLong(args[1]);
        } catch (NumberFormatException e) {
            System.err.println("Invalid -schedule. Expected long [" + args[1] + "]");
            System.exit(-1);
        }

        List<TrafficCamera> cameras = null;

        try {
            cameras = load();
        } catch (IOException e) {
            System.err.println("Unable to load camera data.");
            System.exit(-1);
        }

        System.out.println("There are [" + cameras.size() + "] cameras. Output will be written to [" + dir + "]");
        CountDownLatch latch = new CountDownLatch(cameras.size());
        ApplicationConfiguration configuration = new ApplicationConfiguration(cameras, dir, time);
        CameraDownloader downloader = new CameraDownloader(configuration, latch);
        downloader.run();
        latch.await();
        System.out.println("Done!");
    }
}
