package scs3253034.traffic.camera;

import java.math.BigDecimal;

public class TrafficCamera implements Comparable<TrafficCamera> {

    private static final String BASE_URL = "https://511on.ca/map/Cctv/";

    private String id;
    private BigDecimal latitude;
    private BigDecimal longitude;
    private String url;

    public TrafficCamera(String id, BigDecimal latitude, BigDecimal longitude) {
        super();
        this.id = id;
        this.latitude = latitude;
        this.longitude = longitude;
        this.url = deriveUrl(id);
    }

    @Override
    public int compareTo(TrafficCamera o) {
        return id.compareTo(o.id);
    }

    public String getId() {
        return id;
    }

    public BigDecimal getLatitude() {
        return latitude;
    }

    public BigDecimal getLongitude() {
        return longitude;
    }

    public String getUrl() {
        return url;
    }

    private String deriveUrl(String id) {
        if (id.endsWith("|1")) {
            return BASE_URL + id.replace("|1", "1--1");
        }

        return BASE_URL + id.replace("|", "--");
    }

    @Override
    public String toString() {
        return "TrafficCamera{" +
                "id='" + id + '\'' +
                ", latitude=" + latitude +
                ", longitude=" + longitude +
                ", url=" + getUrl() +
                '}';
    }
}
