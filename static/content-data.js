// Content data for all pages
const contentData = {
  faq: {
    title: "Frequently Asked Questions",
    sections: [
      {
        question: "What is predictive maintenance?",
        answer:
          "Predictive maintenance uses data analysis tools and techniques to detect anomalies and predict when equipment failures might occur, allowing for maintenance to be performed just in time.",
      },
      {
        question: "What types of files do I need to upload?",
        answer:
          "You need to upload two CSV files: one containing temperature/current sensor data and another containing vibration sensor data from your machinery.",
      },
      {
        question: "What is RUL?",
        answer:
          "RUL stands for Remaining Useful Life - it's an estimate of how much longer a machine or component can operate before it needs maintenance or replacement.",
      },
      {
        question: "How accurate are the predictions?",
        answer:
          "Our models are trained on extensive datasets and provide reliable predictions. However, accuracy depends on the quality and consistency of your input data.",
      },
      {
        question: "What models are available?",
        answer:
          "We currently offer two specialized models: Shaft Misalignment Analysis and Bearing Fault Inner (BPFI) Analysis. Each model is trained to detect specific types of failures.",
      },
    ],
  },

  aboutUs: {
    title: "About Us",
    content: `
      <p>We are a team of engineers and data scientists passionate about revolutionizing industrial maintenance through artificial intelligence and machine learning.</p>
      
      <h2>Our Mission</h2>
      <p>To provide accessible, accurate, and actionable predictive maintenance solutions that help industries reduce downtime, minimize costs, and improve operational efficiency.</p>
      
      <h2>Our Team</h2>
      <p>Our multidisciplinary team combines expertise in mechanical engineering, data science, and software development to create cutting-edge predictive maintenance solutions.</p>
      
      <h2>Why Choose Us?</h2>
      <ul>
        <li>Advanced AI models trained on real-world industrial data</li>
        <li>User-friendly interface requiring no technical expertise</li>
        <li>Fast and accurate predictions</li>
        <li>Comprehensive analytics and visualizations</li>
      </ul>
    `,
  },

  aboutModal: {
    title: "About the Model",
    content: `
      <p>Our predictive maintenance models use deep learning techniques to analyze sensor data and predict equipment failures before they occur.</p>
      
      <h2>Model Architecture</h2>
      <p>We use LSTM (Long Short-Term Memory) neural networks that are particularly effective at learning patterns in time-series data. The models process sequential sensor readings to understand equipment behavior over time.</p>
      
      <h2>Key Features</h2>
      <ul>
        <li>Time-domain feature extraction (mean, std, RMS, kurtosis, skewness)</li>
        <li>Frequency-domain analysis using FFT</li>
        <li>Automated preprocessing and resampling</li>
        <li>Sequence-based predictions for improved accuracy</li>
      </ul>
      
      <h2>Two Specialized Models</h2>
      <h3>Shaft Misalignment Model</h3>
      <p>Detects and predicts failures caused by shaft misalignment, a common mechanical issue that can lead to excessive vibration and premature wear.</p>
      
      <h3>BPFI Model</h3>
      <p>Focuses on bearing inner race faults, analyzing specific frequency patterns associated with ball pass frequency on the inner race.</p>
    `,
  },

  modalEvaluation: {
    title: "Model Evaluation",
    content: `
      <p>Rigorous evaluation is crucial for ensuring our models provide reliable predictions in real-world scenarios.</p>
      
      <h2>Evaluation Metrics</h2>
      <p>Our models are evaluated using multiple metrics:</p>
      <ul>
        <li><strong>RMSE (Root Mean Square Error):</strong> Measures prediction accuracy</li>
        <li><strong>MAE (Mean Absolute Error):</strong> Average prediction deviation</li>
        <li><strong>R² Score:</strong> Proportion of variance explained by the model</li>
      </ul>
      
      <h2>Cross-Validation</h2>
      <p>We use time-series cross-validation to ensure models generalize well to unseen data while respecting temporal dependencies.</p>
      
      <h2>Performance Results</h2>
      <p>Both models achieve high accuracy on test datasets, with R² scores above 0.85, demonstrating their reliability for industrial applications.</p>
      
      <h2>Continuous Improvement</h2>
      <p>We continuously monitor model performance and retrain with new data to maintain and improve prediction accuracy.</p>
    `,
  },

  dataCollection: {
    title: "Data Collection",
    content: `
      <p>High-quality data is the foundation of accurate predictive maintenance. Our models require specific types of sensor data to function effectively.</p>
      
      <h2>Required Sensors</h2>
      <h3>Temperature/Current Sensors</h3>
      <p>Monitor thermal and electrical characteristics of machinery. Temperature increases often indicate friction or electrical issues, while current fluctuations can signal motor problems.</p>
      
      <h3>Vibration Sensors</h3>
      <p>Capture mechanical vibrations that reveal equipment health. Different fault types produce characteristic vibration patterns that our models can detect.</p>
      
      <h2>Data Format</h2>
      <p>Data should be provided in CSV format with the following characteristics:</p>
      <ul>
        <li>Original sampling frequency: 25,600 Hz</li>
        <li>Continuous time-series recordings</li>
        <li>Multiple sensor channels</li>
        <li>Optional timestamp column</li>
      </ul>
      
      <h2>Best Practices</h2>
      <ul>
        <li>Ensure sensors are properly calibrated</li>
        <li>Maintain consistent sampling rates</li>
        <li>Collect data under various operating conditions</li>
        <li>Label data with known fault conditions when available</li>
      </ul>
    `,
  },

  maintenanceEvolution: {
    title: "Maintenance Evolution",
    content: `
      <p>Industrial maintenance has evolved significantly over the decades, moving from reactive approaches to sophisticated predictive strategies.</p>
      
      <h2>Reactive Maintenance</h2>
      <p>The traditional "fix it when it breaks" approach. While simple, this method leads to unexpected downtime, higher repair costs, and potential safety hazards.</p>
      
      <h2>Preventive Maintenance</h2>
      <p>Scheduled maintenance based on time intervals or usage metrics. This reduces unexpected failures but often results in unnecessary maintenance and part replacements.</p>
      
      <h2>Condition-Based Maintenance</h2>
      <p>Maintenance triggered by actual equipment condition rather than schedules. Sensors continuously monitor equipment health, and maintenance is performed when certain thresholds are exceeded.</p>
      
      <h2>Predictive Maintenance</h2>
      <p>The current state-of-the-art approach using AI and machine learning to predict failures before they occur. Benefits include:</p>
      <ul>
        <li>Minimized unplanned downtime</li>
        <li>Optimized maintenance schedules</li>
        <li>Extended equipment lifespan</li>
        <li>Reduced maintenance costs</li>
        <li>Improved safety</li>
      </ul>
      
      <h2>Prescriptive Maintenance</h2>
      <p>The next frontier - AI not only predicts when failures will occur but also recommends specific actions to prevent them or optimize maintenance procedures.</p>
    `,
  },

  futureScope: {
    title: "Future Scope",
    content: `
      <p>Predictive maintenance is rapidly evolving, and we're committed to staying at the forefront of innovation.</p>
      
      <h2>Planned Enhancements</h2>
      
      <h3>Additional Fault Types</h3>
      <p>Expanding our model library to detect more types of equipment failures including outer race bearing faults, gear failures, and motor issues.</p>
      
      <h3>Real-Time Monitoring</h3>
      <p>Integration with IoT platforms for continuous, real-time equipment monitoring and instant alerts.</p>
      
      <h3>Multi-Equipment Analysis</h3>
      <p>Analyzing relationships between multiple pieces of equipment to understand cascading failures and system-level health.</p>
      
      <h3>Explainable AI</h3>
      <p>Implementing interpretability features that explain why the model makes certain predictions, building trust and actionable insights.</p>
      
      <h2>Emerging Technologies</h2>
      <ul>
        <li><strong>Edge Computing:</strong> Running models directly on industrial hardware for reduced latency</li>
        <li><strong>Transfer Learning:</strong> Adapting models to new equipment types with minimal training data</li>
        <li><strong>Federated Learning:</strong> Training models across multiple sites while preserving data privacy</li>
        <li><strong>Digital Twins:</strong> Creating virtual replicas of physical equipment for advanced simulation and prediction</li>
      </ul>
      
      <h2>Industry 4.0 Integration</h2>
      <p>Seamless integration with broader Industry 4.0 initiatives, connecting predictive maintenance with supply chain management, production planning, and enterprise resource planning systems.</p>
    `,
  },
};

// Export for use in HTML pages
if (typeof module !== "undefined" && module.exports) {
  module.exports = contentData;
}
