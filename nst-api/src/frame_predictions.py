class FramePredictions:
    def __init__(self, frame_url, image_url, video_name, frame_id, user_data):
        self.user_data = user_data
        self.frame_id = frame_id
        self.frame_url = frame_url
        self.image_url = image_url
        self.video_name = video_name
        self.detected_objects = []

    def reset_detected_object(self):
        self.detected_objects = []

    def add_detected_object(self, detection):
        self.detected_objects.append(detection)

    def create_detected_objects_payload(self, evaulation_server):
        payload = []
        for d_obj in self.detected_objects:
            sub_payload = d_obj.create_payload(evaulation_server)
            payload.append(sub_payload)
        return payload

    def create_payload(self, evaulation_server):
        # TODO(Ahmet): Buradaki ID ve user değeri normalde yok.
        # Alanda sor bakalım bunlar şart mı? Verilen API örneğinde bu kısımlar yok.
        payload = {
            "id": self.frame_id,
            "frame": self.frame_url,
            "detected_objects": self.create_detected_objects_payload(evaulation_server)
        }
        return payload