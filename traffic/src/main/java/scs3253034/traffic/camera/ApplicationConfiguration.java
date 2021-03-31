package scs3253034.traffic.camera;

import java.io.File;
import java.util.List;

public class ApplicationConfiguration {

    private List<TrafficCamera> cameras;
    private File outputDirectory;
    private long schedule;

    public ApplicationConfiguration(List<TrafficCamera> cameras, File outputDirectory, long schedule) {
        super();
        this.cameras = cameras;
        this.schedule = schedule;
        this.outputDirectory = outputDirectory;
    }

    public List<TrafficCamera> getCameras() {
        return cameras;
    }

    public long getSchedule() {
        return schedule;
    }

    public File getOutputDirectory() {
        return outputDirectory;
    }

    @Override
    public String toString() {
        return "ApplicationConfiguration{" +
                "schedule=" + schedule +
                ", outputDirectory=" + outputDirectory +
                '}';
    }
}
