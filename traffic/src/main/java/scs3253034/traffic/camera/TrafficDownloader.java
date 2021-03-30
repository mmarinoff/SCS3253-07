package scs3253034.traffic.camera;

import com.google.maps.DirectionsApi;
import com.google.maps.DirectionsApiRequest;
import com.google.maps.GeoApiContext;
import com.google.maps.errors.ApiException;
import com.google.maps.model.DirectionsResult;
import com.google.maps.model.TrafficModel;
import com.google.maps.model.TravelMode;

import java.io.IOException;

public class TrafficDownloader {

    // TODO : Set up your api code here
    private static final String GOOGLE_API_KEY = "";

    public TrafficDownloader() {
        super();
    }

    public void run() throws InterruptedException, ApiException, IOException {
        GeoApiContext context = new GeoApiContext.Builder()
                .apiKey(GOOGLE_API_KEY)
                .build();

        DirectionsApiRequest request = DirectionsApi.getDirections(context, "43.605645,-79.744697", "43.86936,-78.9015");
        //DirectionsApiRequest request = DirectionsApi.getDirections(context, "M5V3J6" ,"oakville,ON,CA");
        request.departureTimeNow();
        request.mode(TravelMode.DRIVING);
        request.avoid(DirectionsApi.RouteRestriction.TOLLS);
        request.trafficModel(TrafficModel.PESSIMISTIC);
        DirectionsResult result = request.await();

        System.err.println(result);
    }

    public static void main(String[] args) throws Exception {
        TrafficDownloader downloader = new TrafficDownloader();

        downloader.run();
    }
}
