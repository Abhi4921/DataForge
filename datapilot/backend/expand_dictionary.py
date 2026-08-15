"""Generate expanded keyword dictionary with foundational concepts."""
import json

# Load existing dictionary
with open("data/keyword_dictionary.json", "r", encoding="utf-8") as f:
    current = json.load(f)

domains = current["domains"]

# --- ADD FOUNDATIONAL AI/ML CONCEPTS ---
ai_domain = domains.setdefault("artificial_intelligence", {
    "label": "Artificial Intelligence",
    "subdomains": {}
})
foundational_ml = ai_domain["subdomains"].setdefault("foundational_ml", {
    "label": "Foundational ML/AI",
    "concepts": []
})

foundational_concepts = [
    {
        "id": "ai.foundational.machine_learning",
        "canonical_term": "machine learning",
        "synonyms": ["ML", "statistical learning", "learned models", "pattern recognition"],
        "aliases": ["machine-learning"],
        "abbreviations": ["ML"],
        "related_concepts": ["deep learning", "supervised learning", "data science", "artificial intelligence"],
        "potential_ml_tasks": ["classification", "regression", "clustering", "anomaly detection"],
        "dataset_requirements": ["labeled data", "training data", "feature data"],
        "feature_requirements": ["numerical features", "categorical features"],
        "target_requirements": ["target variable", "label column"],
        "importance": 0.98
    },
    {
        "id": "ai.foundational.deep_learning",
        "canonical_term": "deep learning",
        "synonyms": ["DL", "neural network learning", "deep neural networks"],
        "aliases": ["deep-learning"],
        "abbreviations": ["DL"],
        "related_concepts": ["neural networks", "convolutional neural network", "recurrent neural network", "transformer", "machine learning"],
        "potential_ml_tasks": ["classification", "regression", "generation", "detection"],
        "dataset_requirements": ["large datasets", "labeled data"],
        "feature_requirements": ["raw data", "image data", "text data"],
        "target_requirements": ["labels", "annotations"],
        "importance": 0.95
    },
    {
        "id": "ai.foundational.artificial_intelligence",
        "canonical_term": "artificial intelligence",
        "synonyms": ["AI", "intelligent systems", "smart systems"],
        "aliases": ["machine intelligence"],
        "abbreviations": ["AI"],
        "related_concepts": ["machine learning", "deep learning", "natural language processing", "computer vision"],
        "potential_ml_tasks": ["classification", "regression", "prediction", "generation"],
        "dataset_requirements": ["domain data", "labeled data"],
        "feature_requirements": ["relevant features"],
        "target_requirements": ["target variable"],
        "importance": 0.97
    },
    {
        "id": "ai.foundational.data_science",
        "canonical_term": "data science",
        "synonyms": ["data analytics", "predictive analytics", "business analytics"],
        "aliases": [],
        "abbreviations": ["DS"],
        "related_concepts": ["machine learning", "statistics", "data analysis", "data mining"],
        "potential_ml_tasks": ["classification", "regression", "clustering", "analysis"],
        "dataset_requirements": ["structured data", "historical data"],
        "feature_requirements": ["relevant features"],
        "target_requirements": ["target variable"],
        "importance": 0.93
    },
    {
        "id": "ai.foundational.supervised_learning",
        "canonical_term": "supervised learning",
        "synonyms": ["labeled learning", "guided learning"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["classification", "regression", "machine learning"],
        "potential_ml_tasks": ["classification", "regression"],
        "dataset_requirements": ["labeled examples", "training data with labels"],
        "feature_requirements": ["input features"],
        "target_requirements": ["target labels", "output variable"],
        "importance": 0.9
    },
    {
        "id": "ai.foundational.unsupervised_learning",
        "canonical_term": "unsupervised learning",
        "synonyms": ["self-organized learning", "unlabeled learning"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["clustering", "anomaly detection", "dimensionality reduction"],
        "potential_ml_tasks": ["clustering", "anomaly detection", "dimensionality reduction"],
        "dataset_requirements": ["unlabeled data", "feature data"],
        "feature_requirements": ["input features"],
        "target_requirements": [],
        "importance": 0.88
    },
    {
        "id": "ai.foundational.transfer_learning",
        "canonical_term": "transfer learning",
        "synonyms": ["domain adaptation", "fine-tuning", "pre-trained models"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["deep learning", "neural networks", "fine tuning"],
        "potential_ml_tasks": ["classification", "detection", "generation"],
        "dataset_requirements": ["pre-trained model", "target domain data"],
        "feature_requirements": ["domain-specific features"],
        "target_requirements": ["target labels"],
        "importance": 0.85
    },
    {
        "id": "ai.foundational.feature_engineering",
        "canonical_term": "feature engineering",
        "synonyms": ["feature extraction", "feature creation", "feature selection"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["data preprocessing", "dimensionality reduction", "feature selection"],
        "potential_ml_tasks": ["feature engineering", "feature selection"],
        "dataset_requirements": ["raw data", "domain knowledge"],
        "feature_requirements": ["raw features"],
        "target_requirements": ["engineered features"],
        "importance": 0.85
    },
    {
        "id": "ai.foundational.data_preprocessing",
        "canonical_term": "data preprocessing",
        "synonyms": ["data cleaning", "data preparation", "data transformation"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["feature engineering", "data augmentation", "missing value imputation"],
        "potential_ml_tasks": ["data cleaning", "data transformation"],
        "dataset_requirements": ["raw data", "messy data"],
        "feature_requirements": ["raw features"],
        "target_requirements": [],
        "importance": 0.82
    },
    {
        "id": "ai.foundational.model_evaluation",
        "canonical_term": "model evaluation",
        "synonyms": ["model testing", "performance evaluation", "model assessment"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["cross-validation", "accuracy", "precision", "recall"],
        "potential_ml_tasks": ["evaluation", "benchmarking"],
        "dataset_requirements": ["test data", "validation data"],
        "feature_requirements": ["test features"],
        "target_requirements": ["test labels"],
        "importance": 0.8
    },
    {
        "id": "ai.foundational.prediction",
        "canonical_term": "prediction",
        "synonyms": ["forecasting", "projecting", "estimation"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["regression", "time series forecasting", "classification"],
        "potential_ml_tasks": ["regression", "classification", "forecasting"],
        "dataset_requirements": ["historical data", "labeled data"],
        "feature_requirements": ["input features"],
        "target_requirements": ["target values"],
        "importance": 0.9
    },
    {
        "id": "ai.foundational.detection",
        "canonical_term": "detection",
        "synonyms": ["identification", "recognition", "discovery"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["anomaly detection", "object detection", "classification"],
        "potential_ml_tasks": ["binary classification", "detection", "recognition"],
        "dataset_requirements": ["labeled data", "positive and negative examples"],
        "feature_requirements": ["discriminative features"],
        "target_requirements": ["presence absence labels"],
        "importance": 0.88
    },
    {
        "id": "ai.foundational.recognition",
        "canonical_term": "recognition",
        "synonyms": ["identification", "detection", "classification"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["image classification", "object detection", "speech recognition"],
        "potential_ml_tasks": ["classification", "detection"],
        "dataset_requirements": ["labeled data", "categorized examples"],
        "feature_requirements": ["discriminative features"],
        "target_requirements": ["category labels"],
        "importance": 0.85
    },
    {
        "id": "ai.foundational.optimization",
        "canonical_term": "optimization",
        "synonyms": ["improvement", "efficiency", "minimization", "maximization"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["resource allocation", "scheduling optimization", "operations research"],
        "potential_ml_tasks": ["optimization", "regression"],
        "dataset_requirements": ["constraint data", "objective function data"],
        "feature_requirements": ["decision variables"],
        "target_requirements": ["optimal values"],
        "importance": 0.82
    },
    {
        "id": "ai.foundational.forecasting",
        "canonical_term": "forecasting",
        "synonyms": ["prediction", "projection", "future estimation"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["time series forecasting", "regression", "demand forecasting"],
        "potential_ml_tasks": ["regression", "time series forecasting"],
        "dataset_requirements": ["historical time series", "temporal data"],
        "feature_requirements": ["time features", "lag features"],
        "target_requirements": ["future values"],
        "importance": 0.88
    },
    {
        "id": "ai.foundational.recommendation",
        "canonical_term": "recommendation",
        "synonyms": ["suggestion", "personalization", "ranking"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["collaborative filtering", "content based filtering", "product recommendation"],
        "potential_ml_tasks": ["recommendation", "ranking", "collaborative filtering"],
        "dataset_requirements": ["user item interactions", "user preferences"],
        "feature_requirements": ["user features", "item features"],
        "target_requirements": ["ratings", "engagement"],
        "importance": 0.85
    },
]

foundational_ml["concepts"].extend(foundational_concepts)

# --- ADD IoT FOUNDATIONAL CONCEPTS ---
iot_domain = domains.setdefault("iot", {
    "label": "Internet of Things (IoT)",
    "subdomains": {}
})
iot_foundational = iot_domain["subdomains"].setdefault("iot_foundational", {
    "label": "IoT Foundational",
    "concepts": []
})

iot_concepts = [
    {
        "id": "iot.foundational.internet_of_things",
        "canonical_term": "internet of things",
        "synonyms": ["IoT", "connected devices", "smart devices", "wireless sensor networks"],
        "aliases": ["machine to machine", "M2M"],
        "abbreviations": ["IoT"],
        "related_concepts": ["sensor data", "edge computing", "smart cities", "iot devices"],
        "potential_ml_tasks": ["anomaly detection", "classification", "time series forecasting"],
        "dataset_requirements": ["sensor readings", "device telemetry"],
        "feature_requirements": ["sensor values", "device identifiers", "timestamps"],
        "target_requirements": ["device status", "anomaly labels"],
        "importance": 0.92
    },
    {
        "id": "iot.foundational.iot_devices",
        "canonical_term": "iot devices",
        "synonyms": ["smart devices", "connected devices", "sensor nodes", "edge devices"],
        "aliases": ["iot device"],
        "abbreviations": [],
        "related_concepts": ["internet of things", "sensor data", "edge computing"],
        "potential_ml_tasks": ["classification", "anomaly detection", "prediction"],
        "dataset_requirements": ["device logs", "telemetry data"],
        "feature_requirements": ["device type", "firmware version", "network info"],
        "target_requirements": ["device status", "failure labels"],
        "importance": 0.88
    },
    {
        "id": "iot.foundational.sensor_data",
        "canonical_term": "sensor data",
        "synonyms": ["telemetry", "sensor readings", "measurements", "instrument readings"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["iot devices", "time series", "anomaly detection"],
        "potential_ml_tasks": ["regression", "classification", "anomaly detection"],
        "dataset_requirements": ["sensor logs", "measurement data"],
        "feature_requirements": ["sensor values", "timestamps", "sensor type"],
        "target_requirements": ["measurement targets", "anomaly labels"],
        "importance": 0.85
    },
    {
        "id": "iot.foundational.edge_computing",
        "canonical_term": "edge computing",
        "synonyms": ["fog computing", "on device processing", "distributed computing"],
        "aliases": [],
        "abbreviations": [],
        "related_concepts": ["iot devices", "real time processing", "embedded systems"],
        "potential_ml_tasks": ["classification", "detection"],
        "dataset_requirements": ["device data", "resource constraints"],
        "feature_requirements": ["compute resources", "latency requirements"],
        "target_requirements": ["processing decisions"],
        "importance": 0.8
    },
]

iot_foundational["concepts"].extend(iot_concepts)

# --- ADD NETWORK/SECURITY BROADER TERMS ---
cyber_domain = domains["cybersecurity"]
net_security = cyber_domain["subdomains"].setdefault("network_security_foundational", {
    "label": "Network Security Foundational",
    "concepts": []
})

net_concepts = [
    {
        "id": "cyber.net_foundational.network_security",
        "canonical_term": "network security",
        "synonyms": ["cyber defense", "network defense", "information security", "cybersecurity"],
        "aliases": ["network defense"],
        "abbreviations": ["NETSEC"],
        "related_concepts": ["intrusion detection", "firewall", "network traffic analysis"],
        "potential_ml_tasks": ["binary classification", "multi class classification", "anomaly detection"],
        "dataset_requirements": ["network logs", "security events"],
        "feature_requirements": ["packet features", "flow features"],
        "target_requirements": ["threat labels", "benign malicious labels"],
        "importance": 0.92
    },
    {
        "id": "cyber.net_foundational.network_traffic",
        "canonical_term": "network traffic",
        "synonyms": ["packet data", "network flow", "traffic data", "network packets"],
        "aliases": ["traffic flow"],
        "abbreviations": [],
        "related_concepts": ["network traffic analysis", "intrusion detection", "network security"],
        "potential_ml_tasks": ["classification", "clustering", "anomaly detection"],
        "dataset_requirements": ["packet captures", "flow records", "netflow data"],
        "feature_requirements": ["packet sizes", "flow durations", "protocol types"],
        "target_requirements": ["traffic labels", "attack types"],
        "importance": 0.9
    },
    {
        "id": "cyber.net_foundational.abnormal_behavior",
        "canonical_term": "abnormal behavior",
        "synonyms": ["anomalous behavior", "malicious behavior", "suspicious activity", "threat behavior"],
        "aliases": ["suspicious behavior"],
        "abbreviations": [],
        "related_concepts": ["anomaly detection", "intrusion detection", "malware detection"],
        "potential_ml_tasks": ["anomaly detection", "binary classification"],
        "dataset_requirements": ["behavior logs", "activity data"],
        "feature_requirements": ["behavioral features", "activity patterns"],
        "target_requirements": ["normal abnormal labels"],
        "importance": 0.88
    },
    {
        "id": "cyber.net_foundational.malicious_activity",
        "canonical_term": "malicious activity",
        "synonyms": ["threat detection", "attack detection", "hostile activity", "adversarial activity"],
        "aliases": ["attack activity"],
        "abbreviations": [],
        "related_concepts": ["intrusion detection", "anomaly detection", "malware detection"],
        "potential_ml_tasks": ["binary classification", "multi class classification"],
        "dataset_requirements": ["security logs", "attack data"],
        "feature_requirements": ["activity features", "context features"],
        "target_requirements": ["attack labels", "threat levels"],
        "importance": 0.88
    },
    {
        "id": "cyber.net_foundational.intrusion_detection_system",
        "canonical_term": "intrusion detection system",
        "synonyms": ["IDS", "network intrusion detection system", "security monitoring system"],
        "aliases": ["intrusion detector"],
        "abbreviations": ["IDS", "NIDS"],
        "related_concepts": ["intrusion detection", "network security", "anomaly detection"],
        "potential_ml_tasks": ["binary classification", "multi class classification"],
        "dataset_requirements": ["network traffic data", "attack signatures"],
        "feature_requirements": ["flow features", "packet features"],
        "target_requirements": ["attack benign labels"],
        "importance": 0.88
    },
]

net_security["concepts"].extend(net_concepts)

# --- ADD CYBERSECURITY BROADER FRAUD ---
cyber_fraud = cyber_domain["subdomains"].setdefault("fraud_foundational", {
    "label": "Fraud Foundational",
    "concepts": []
})

fraud_concepts = [
    {
        "id": "cyber.fraud_foundational.financial_fraud",
        "canonical_term": "financial fraud",
        "synonyms": ["fraud", "payment fraud", "monetary fraud", "financial crime"],
        "aliases": ["money fraud"],
        "abbreviations": [],
        "related_concepts": ["fraud detection", "anomaly detection", "credit card fraud"],
        "potential_ml_tasks": ["binary classification", "anomaly detection"],
        "dataset_requirements": ["transaction records", "fraud labels"],
        "feature_requirements": ["transaction amount", "transaction time", "merchant info"],
        "target_requirements": ["fraud legitimate labels"],
        "importance": 0.9
    },
    {
        "id": "cyber.fraud_foundational.credit_card",
        "canonical_term": "credit card",
        "synonyms": ["payment card", "card transaction", "credit card transaction"],
        "aliases": ["debit card"],
        "abbreviations": [],
        "related_concepts": ["financial fraud", "transaction fraud", "payment processing"],
        "potential_ml_tasks": ["binary classification", "fraud detection"],
        "dataset_requirements": ["transaction data", "card transaction history"],
        "feature_requirements": ["amount", "merchant", "location", "time"],
        "target_requirements": ["fraud labels"],
        "importance": 0.85
    },
    {
        "id": "cyber.fraud_foundational.credit_card_fraud",
        "canonical_term": "credit card fraud",
        "synonyms": ["card fraud", "payment card fraud", "credit card transaction fraud"],
        "aliases": ["credit card fraud detection"],
        "abbreviations": [],
        "related_concepts": ["financial fraud", "fraud detection", "anomaly detection"],
        "potential_ml_tasks": ["binary classification", "anomaly detection", "imbalanced classification"],
        "dataset_requirements": ["credit card transactions", "fraud labels"],
        "feature_requirements": ["transaction amount", "merchant", "location", "time", "card features"],
        "target_requirements": ["fraud legitimate labels"],
        "importance": 0.9
    },
]

cyber_fraud["concepts"].extend(fraud_concepts)

# --- ADD EDUCATION BROADER CONCEPTS ---
edu_domain = domains["education"]
edu_analytics = edu_domain["subdomains"].setdefault("education_foundational", {
    "label": "Education Foundational",
    "concepts": []
})

edu_concepts = [
    {
        "id": "edu.foundational.education",
        "canonical_term": "education",
        "synonyms": ["academic", "teaching", "schooling"],
        "aliases": ["educational"],
        "abbreviations": [],
        "related_concepts": ["student performance", "learning analytics", "educational data mining"],
        "potential_ml_tasks": ["classification", "regression", "recommendation"],
        "dataset_requirements": ["student records", "academic data"],
        "feature_requirements": ["student features", "course features"],
        "target_requirements": ["academic outcomes"],
        "importance": 0.9
    },
    {
        "id": "edu.foundational.student_performance",
        "canonical_term": "student performance",
        "synonyms": ["academic performance", "student achievement", "academic outcome", "student outcomes", "learning outcomes"],
        "aliases": ["student academic performance"],
        "abbreviations": [],
        "related_concepts": ["student performance prediction", "educational data mining", "learning analytics"],
        "potential_ml_tasks": ["regression", "classification"],
        "dataset_requirements": ["student records", "academic data", " grades"],
        "feature_requirements": ["attendance", "scores", "grades", "study hours"],
        "target_requirements": ["GPA", "pass fail", "grade"],
        "importance": 0.92
    },
    {
        "id": "edu.foundational.attendance",
        "canonical_term": "attendance",
        "synonyms": ["class attendance", "attendance tracking", "class presence", "participation"],
        "aliases": ["school attendance"],
        "abbreviations": [],
        "related_concepts": ["student performance", "engagement", "participation"],
        "potential_ml_tasks": ["regression", "classification"],
        "dataset_requirements": ["attendance records", "class schedules"],
        "feature_requirements": ["attendance dates", "absence count", "class type"],
        "target_requirements": ["attendance rate", "present absent"],
        "importance": 0.85
    },
    {
        "id": "edu.foundational.examination",
        "canonical_term": "examination",
        "synonyms": ["exam", "assessment", "test", "quiz", "evaluation"],
        "aliases": ["exam scores"],
        "abbreviations": [],
        "related_concepts": ["test scores", "academic assessment", "student performance"],
        "potential_ml_tasks": ["regression", "classification"],
        "dataset_requirements": ["exam results", "test scores"],
        "feature_requirements": ["exam scores", "question responses"],
        "target_requirements": ["final grade", "pass fail"],
        "importance": 0.85
    },
    {
        "id": "edu.foundational.study_hours",
        "canonical_term": "study hours",
        "synonyms": ["study time", "learning time", "homework hours", "revision time"],
        "aliases": ["hours studied"],
        "abbreviations": [],
        "related_concepts": ["student performance", "engagement", "time on task"],
        "potential_ml_tasks": ["regression", "correlation analysis"],
        "dataset_requirements": ["student activity logs", "time tracking data"],
        "feature_requirements": ["hours per subject", "study patterns"],
        "target_requirements": ["performance outcomes"],
        "importance": 0.82
    },
    {
        "id": "edu.foundational.class_participation",
        "canonical_term": "class participation",
        "synonyms": ["participation", "class engagement", "student engagement", "class interaction"],
        "aliases": ["class participation"],
        "abbreviations": [],
        "related_concepts": ["student performance", "engagement", "attendance"],
        "potential_ml_tasks": ["regression", "classification"],
        "dataset_requirements": ["participation records", "engagement data"],
        "feature_requirements": ["participation count", "interaction type"],
        "target_requirements": ["performance outcomes"],
        "importance": 0.82
    },
    {
        "id": "edu.foundational.assignment_scores",
        "canonical_term": "assignment scores",
        "synonyms": ["homework scores", "assignment grades", "coursework marks", "assignment results"],
        "aliases": ["homework marks"],
        "abbreviations": [],
        "related_concepts": ["student performance", "assessment", "academic grading"],
        "potential_ml_tasks": ["regression", "classification"],
        "dataset_requirements": ["assignment submissions", "graded assignments"],
        "feature_requirements": ["assignment type", "submission time", "score"],
        "target_requirements": ["grade", "pass fail"],
        "importance": 0.82
    },
]

edu_analytics["concepts"].extend(edu_concepts)

# --- ADD ENERGY BROADER CONCEPTS ---
energy_domain = domains.get("energy", {"label": "Energy", "subdomains": {}})
if "energy" not in domains:
    domains["energy"] = energy_domain
energy_found = energy_domain["subdomains"].setdefault("energy_foundational", {
    "label": "Energy Foundational",
    "concepts": []
})

energy_concepts = [
    {
        "id": "energy.foundational.energy_consumption",
        "canonical_term": "energy consumption",
        "synonyms": ["electricity usage", "power consumption", "energy usage", "electricity consumption"],
        "aliases": ["energy use"],
        "abbreviations": [],
        "related_concepts": ["energy demand forecasting", "smart grid", "load forecasting"],
        "potential_ml_tasks": ["regression", "time series forecasting"],
        "dataset_requirements": ["energy usage data", "smart meter data"],
        "feature_requirements": ["time of day", "temperature", "building type"],
        "target_requirements": ["energy consumption values"],
        "importance": 0.9
    },
    {
        "id": "energy.foundational.electricity",
        "canonical_term": "electricity",
        "synonyms": ["electric power", "power", "electric energy"],
        "aliases": ["electric"],
        "abbreviations": [],
        "related_concepts": ["energy consumption", "power grid", "smart grid"],
        "potential_ml_tasks": ["regression", "forecasting"],
        "dataset_requirements": ["power generation data", "grid data"],
        "feature_requirements": ["generation capacity", "demand"],
        "target_requirements": ["power output"],
        "importance": 0.85
    },
    {
        "id": "energy.foundational.energy_forecasting",
        "canonical_term": "energy forecasting",
        "synonyms": ["energy prediction", "power forecasting", "demand forecasting"],
        "aliases": ["energy demand prediction"],
        "abbreviations": [],
        "related_concepts": ["energy demand forecasting", "load forecasting", "time series forecasting"],
        "potential_ml_tasks": ["regression", "time series forecasting"],
        "dataset_requirements": ["historical energy data", "weather data"],
        "feature_requirements": ["time features", "weather", "historical usage"],
        "target_requirements": ["future energy values"],
        "importance": 0.85
    },
]

energy_found["concepts"].extend(energy_concepts)

# --- FIX RELATED CONCEPTS: replace underscores with spaces ---
for domain_data in domains.values():
    for sub_data in domain_data.get("subdomains", {}).values():
        for concept in sub_data.get("concepts", []):
            concept["related_concepts"] = [
                r.replace("_", " ") for r in concept.get("related_concepts", [])
            ]
            concept["dataset_requirements"] = [
                r.replace("_", " ") for r in concept.get("dataset_requirements", [])
            ]
            concept["feature_requirements"] = [
                r.replace("_", " ") for r in concept.get("feature_requirements", [])
            ]
            concept["target_requirements"] = [
                r.replace("_", " ") for r in concept.get("target_requirements", [])
            ]

# Update metadata
total = sum(
    len(s.get("concepts", []))
    for dom in domains.values()
    for s in dom.get("subdomains", {}).values()
)
current["metadata"]["version"] = "0.2.0"
current["metadata"]["total_concepts"] = total
current["domains"] = domains

with open("data/keyword_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(current, f, indent=2, ensure_ascii=False)

print(f"Dictionary updated: {total} concepts across {len(domains)} domains")
