package scs3253034.traffic.camera;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import org.asynchttpclient.AsyncCompletionHandler;
import org.asynchttpclient.AsyncHandler;
import org.asynchttpclient.AsyncHttpClient;
import org.asynchttpclient.BoundRequestBuilder;
import org.asynchttpclient.Dsl;
import org.asynchttpclient.HttpResponseBodyPart;
import org.asynchttpclient.Response;
import org.springframework.util.StringUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;

public final class CameraListDownloader {

    private static final String URL = "https://511on.ca/map/mapIcons/Cameras";
    private static final ObjectMapper mapper = new ObjectMapper();

    public CameraListDownloader() {
        super();
    }

    public static List<TrafficCamera> download() throws Exception {
        AsyncHttpClient client = Dsl.asyncHttpClient();
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        CountDownLatch latch = new CountDownLatch(1);
        BoundRequestBuilder builder = client.prepareGet(URL);

        builder.execute(new AsyncCompletionHandler<ByteArrayOutputStream>() {

            @Override
            public AsyncHandler.State onBodyPartReceived(HttpResponseBodyPart bodyPart)
                    throws Exception {
                stream.write(bodyPart.getBodyPartBytes());
                return State.CONTINUE;
            }

            @Override
            public ByteArrayOutputStream onCompleted(Response response) {
                latch.countDown();
                return stream;
            }
        });

        latch.await();
        stream.close();
        client.close();

        List<TrafficCamera> result = new ArrayList<>();
        JsonNode rootNode = mapper.readTree(stream.toString());
        ArrayNode items = (ArrayNode) rootNode.get("item2");

        if (items != null) {
           for (int i = 0; i < items.size(); i++) {
               JsonNode node = items.get(i);
               String id = node.get("itemId").asText();

               if (StringUtils.hasLength(id)) {
                   ArrayNode location = (ArrayNode) node.get("location");

                   if (location != null && location.size() == 2) {
                       try {
                           BigDecimal latitude = new BigDecimal(location.get(0).asText());
                           BigDecimal longitude = new BigDecimal(location.get(1).asText());
                           result.add(new TrafficCamera(id, latitude, longitude));
                       } catch (Exception e) {
                           // Ignore and move on
                       }
                   }
               }
           }
        }

        return result;
    }

    public static void setup(BoundRequestBuilder builder) {
        builder.addHeader("content-type", "application/json");
    }

    public static void main(String[] args) throws Exception {
        List<TrafficCamera> cameras = download();
        Collections.sort(cameras);
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File("./src/main/resources/camera-list.json"), cameras);
    }
}
