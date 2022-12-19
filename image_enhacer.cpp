#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  // Load the input image
  cv::Mat image = cv::imread("input.jpg");

  // Check if the image was loaded successfully
  if (image.empty()) {
    std::cout << "Could not load input image!" << std::endl;
    return -1;
  }

  // Convert the image to grayscale
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

  // Resize the image to fit the model's input size
  cv::Mat image_resized;
  cv::resize(image_gray, image_resized, cv::Size(256, 256));

  // Convert the image to a tensor
  tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 256, 256, 1}));
  auto tensor_map = image_tensor.tensor<float, 4>();
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      tensor_map(0, y, x, 0) = static_cast<float>(image_resized.at<uchar>(y, x)) / 255.0;
    }
  }

  // Load the model
  tensorflow::GraphDef graph_def;
  tensorflow::SessionOptions session_options;
  std::unique_ptr<tensorflow::Session> session = tensorflow::NewSession(session_options);
  tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "model.pb", &graph_def);
  if (!status.ok()) {
    std::cout << "Could not load model: " << status.ToString() << std::endl;
    return -1;
  }
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << "Could not create session: " << status.ToString() << std::endl;
    return -1;
  }

  // Run the model
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{"input_1", image_tensor}}, {"output_1"}, {}, &outputs);
  if (!run_status.ok()) {
    std::cout << "Could not run model: " << run_status.ToString() << std::endl;
    return -1;
  }

  // Extract the enhanced image from the output tensor
  cv::Mat image_enhanced(256, 256, CV_8UC1);
  auto output_map = outputs[0]..tensor<float, 4>();
for (int y = 0; y < 256; ++y) {
for (int x = 0; x < 256; ++x) {
image_enhanced.at<uchar>(y, x) = static_cast<uchar>(output_map(0, y, x, 0) * 255.0);
}
}

// Save the enhanced image
cv::imwrite("output.jpg", image_enhanced);

return 0;
}
