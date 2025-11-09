# =============================================================================
# مدل پیشرفته هوش آینده‌نگر - کد یکپارچه
# Advanced Anticipatory Intelligence Model - Integrated Code
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns

# تنظیمات نمایش
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# کلاس‌های مدل
# =============================================================================

class PredictiveModel:
    """مدل پیش‌بینی پیشرفته"""
    
    def __init__(self):
        self.models = {}
        
    def fit_ensemble(self, X_train, y_train):
        """آموزش مدل‌های ترکیبی برای پیش‌بینی"""
        # مدل جنگل تصادفی
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # مدل شبکه عصبی
        nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=1000)
        nn_model.fit(X_train, y_train)
        
        self.models = {'random_forest': rf_model, 'neural_network': nn_model}
    
    def probabilistic_forecast(self, X_current, steps=5):
        """پیش‌بینی احتمالی با فاصله اطمینان"""
        forecasts = {}
        
        for name, model in self.models.items():
            predictions = []
            X_temp = X_current.copy()
            
            for step in range(steps):
                if len(X_temp.shape) == 1:
                    X_temp = X_temp.reshape(1, -1)
                pred = model.predict(X_temp)[0]
                predictions.append(pred)
                # به‌روزرسانی حالت برای قدم بعدی
                X_temp = self.update_state(X_temp, pred)
            
            forecasts[name] = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'predictions': predictions
            }
        
        return forecasts
    
    def update_state(self, state, prediction):
        """به‌روزرسانی حالت سیستم"""
        new_state = np.roll(state, -1)
        if len(new_state.shape) == 1:
            new_state[-1] = prediction
        else:
            new_state[0, -1] = prediction
        return new_state

class TraditionalIntelligenceModel:
    """مدل اطلاعاتی سنتی (واکنشی)"""
    
    def predict(self, data):
        """پیش‌بینی بر اساس الگوهای تاریخی"""
        if len(data) > 10:
            recent_data = data.tail(10)
        else:
            recent_data = data
        return recent_data['threat_level'].mean()
    
    def predict_proba(self, data):
        """پیش‌بینی احتمالی"""
        base_prob = self.predict(data)
        # اضافه کردن نویز کوچک برای شبیه‌سازی عدم قطعیت
        return max(0, min(1, base_prob + np.random.normal(0, 0.1)))

class AdvancedAnticipatoryModel:
    """مدل پیشرفته آینده‌نگر"""
    
    def __init__(self):
        self.predictive_model = PredictiveModel()
        self.risk_threshold = 0.6
        self.is_trained = False
        
    def predict_proba(self, data):
        """پیش‌بینی احتمالی با درنظرگیری شاخص‌های پیش‌نگر"""
        
        # استخراج ویژگی‌های پیش‌نگر
        features = self.extract_anticipatory_features(data)
        
        if not self.is_trained and len(data) > 30:
            self.train_model(data)
        
        if self.is_trained:
            # پیش‌بینی با مدل ترکیبی
            forecasts = self.predictive_model.probabilistic_forecast(features)
            
            # ترکیب پیش‌بینی‌ها
            combined_risk = np.mean([
                forecasts['random_forest']['mean'],
                forecasts['neural_network']['mean']
            ])
        else:
            # حالت سقوط به مدل سنتی
            combined_risk = np.mean(features[:2])
        
        return max(0, min(1, combined_risk))
    
    def train_model(self, data):
        """آموزش مدل پیش‌بینی"""
        try:
            X_train = []
            y_train = []
            
            for i in range(10, len(data)-5):
                features = self.extract_anticipatory_features(data.iloc[:i])
                target = data['threat_level'].iloc[i+5]  # پیش‌بینی ۵ قدم جلوتر
                X_train.append(features)
                y_train.append(target)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            self.predictive_model.fit_ensemble(X_train, y_train)
            self.is_trained = True
            print("✅ مدل آینده‌نگر آموزش داده شد")
        except Exception as e:
            print(f"⚠️ خطا در آموزش مدل: {e}")
    
    def extract_anticipatory_features(self, data):
        """استخراج شاخص‌های پیش‌نگر از داده‌ها"""
        
        features = []
        
        if len(data) < 5:
            return np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        # ۱. نرخ تغییر شتاب (سیگنال ضعیف)
        threat_series = data['threat_level'].values
        if len(threat_series) > 2:
            first_deriv = np.gradient(threat_series)
            second_deriv = np.gradient(first_deriv)
            features.append(np.mean(second_deriv[-3:]) if len(second_deriv) >= 3 else 0)
        else:
            features.append(0)
        
        # ۲. نوسانات غیرعادی
        if 'indicators' in data.columns:
            if hasattr(data['indicators'].iloc[0], '__len__'):
                recent_indicators = np.vstack(data['indicators'].values[-5:])
                volatility = np.std(recent_indicators, axis=0).mean()
            else:
                volatility = data['indicators'].tail(5).std()
        else:
            volatility = 0.1
        features.append(volatility)
        
        # ۳. روند تغییرات
        if len(threat_series) > 5:
            trend = np.polyfit(range(len(threat_series[-5:])), threat_series[-5:], 1)[0]
        else:
            trend = 0
        features.append(trend)
        
        # ۴. میانگین متحرک
        moving_avg = np.mean(threat_series[-5:]) if len(threat_series) >= 5 else np.mean(threat_series)
        features.append(moving_avg)
        
        # ۵. انحراف معیار
        threat_std = np.std(threat_series[-5:]) if len(threat_series) >= 5 else 0.1
        features.append(threat_std)
        
        return np.array(features)

class AnticipatoryIntelligenceSystem:
    """سیستم یکپارچه هوش آینده‌نگر"""
    
    def __init__(self):
        self.traditional_model = TraditionalIntelligenceModel()
        self.anticipatory_model = AdvancedAnticipatoryModel()
        self.performance_history = []
    
    def simulate_data(self, n_steps=200):
        """تولید داده‌های شبیه‌سازی شده"""
        np.random.seed(42)
        
        time_index = pd.date_range('2024-01-01', periods=n_steps, freq='D')
        
        # ایجاد الگوهای پیچیده در داده‌ها
        base_pattern = np.sin(2 * np.pi * np.arange(n_steps) / 50)
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(n_steps) / 20)
        noise = 0.2 * np.random.normal(0, 1, n_steps)
        
        # سطح تهدید پایه با الگوهای پیچیده
        threat_base = 0.3 + 0.2 * base_pattern + 0.1 * seasonal + 0.1 * noise
        
        # اضافه کردن رویدادهای تهدیدآمیز تصادفی
        threat_events = np.zeros(n_steps)
        event_indices = np.random.choice(n_steps-10, size=15, replace=False)
        for idx in event_indices:
            threat_events[idx:idx+5] = 0.8 + 0.2 * np.random.random()
        
        threat_level = np.clip(threat_base + threat_events, 0, 1)
        
        # شاخص‌های مختلف
        indicators = np.random.normal(0, 1, (n_steps, 5))
        
        data = pd.DataFrame({
            'timestamp': time_index,
            'threat_level': threat_level,
            'indicators': list(indicators),
            'context_1': np.random.uniform(0, 1, n_steps),
            'context_2': np.random.uniform(0, 1, n_steps),
            'context_3': np.random.uniform(0, 1, n_steps)
        })
        
        return data

# =============================================================================
# توابع ارزیابی
# =============================================================================

def calculate_early_detection(actual, predicted, window=5):
    """محاسبه نرخ شناسایی زودهنگام"""
    early_detections = 0
    total_threats = 0
    
    actual_array = actual if isinstance(actual, np.ndarray) else actual.values
    
    for i in range(len(predicted)):
        if i < len(actual_array) and actual_array[i] > 0.7:  # تهدید واقعی
            total_threats += 1
            # بررسی آیا در پنجره قبلی هشدار داده بود
            for j in range(max(0, i-window), i):
                if j < len(predicted) and predicted[j] > 0.6:
                    early_detections += 1
                    break
    
    return early_detections / total_threats if total_threats > 0 else 0

def calculate_false_positive(actual, predicted):
    """محاسبه نرخ هشدارهای کاذب"""
    actual_binary = (actual > 0.7).astype(int)
    predicted_binary = (np.array(predicted) > 0.6).astype(int)
    
    fp = np.sum((predicted_binary == 1) & (actual_binary == 0))
    total_negatives = np.sum(actual_binary == 0)
    
    return fp / total_negatives if total_negatives > 0 else 0

def evaluate_models(results, actual):
    """ارزیابی جامع عملکرد مدل‌ها"""
    
    metrics = {}
    actual_array = actual.values if isinstance(actual, pd.Series) else actual
    
    for model_name, predictions in results.items():
        if model_name == 'actual':
            continue
            
        pred_array = np.array(predictions)
        actual_subset = actual_array[-len(pred_array):]
        
        pred_binary = (pred_array > 0.5).astype(int)
        actual_binary = (actual_subset > 0.5).astype(int)
        
        # محاسبه معیارها
        precision = precision_score(actual_binary, pred_binary, zero_division=0)
        recall = recall_score(actual_binary, pred_binary, zero_division=0)
        f1 = f1_score(actual_binary, pred_binary, zero_division=0)
        
        metrics[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': roc_auc_score(actual_binary, pred_array),
            'early_detection_rate': calculate_early_detection(actual_subset, pred_array),
            'false_positive_rate': calculate_false_positive(actual_subset, pred_array)
        }
    
    return metrics

def plot_comparison(results, metrics, data):
    """ترسیم نمودارهای مقایسه‌ای"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # نمودار ۱: مقایسه پیش‌بینی‌ها در طول زمان
    time_points = range(len(results['traditional']))
    actual_values = data['threat_level'].values[-len(results['traditional']):]
    
    ax1.plot(time_points, results['traditional'], 'r-', label='مدل سنتی', alpha=0.7, linewidth=2)
    ax1.plot(time_points, results['anticipatory'], 'b-', label='مدل آینده‌نگر', alpha=0.7, linewidth=2)
    ax1.plot(time_points, actual_values, 'g--', label='واقعیت', alpha=0.7, linewidth=1)
    ax1.set_title('مقایسه پیش‌بینی تهدیدات در طول زمان', fontsize=14, fontweight='bold')
    ax1.set_xlabel('زمان')
    ax1.set_ylabel('سطح تهدید')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # نمودار ۲: معیارهای عملکرد
    model_names = list(metrics.keys())
    performance_metrics = ['precision', 'recall', 'f1_score']
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(performance_metrics):
        values = [metrics[model][metric] for model in model_names]
        ax2.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)
    
    ax2.set_title('مقایسه معیارهای عملکرد', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # نمودار ۳: شناسایی زودهنگام و هشدارهای کاذب
    early_detection = [metrics[model]['early_detection_rate'] for model in model_names]
    false_positive = [metrics[model]['false_positive_rate'] for model in model_names]
    
    x = np.arange(len(model_names))
    ax3.bar(x - 0.2, early_detection, 0.4, label='شناسایی زودهنگام', alpha=0.8)
    ax3.bar(x + 0.2, false_positive, 0.4, label='هشدار کاذب', alpha=0.8)
    ax3.set_title('شناسایی زودهنگام vs هشدارهای کاذب', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # نمودار ۴: منحنی ROC
    for model_name in model_names:
        if model_name in results:
            fpr, tpr, _ = roc_curve(
                data['threat_level'].values[-len(results[model_name]):] > 0.5,
                results[model_name]
            )
            ax4.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics[model_name]["auc_roc"]:.3f})', linewidth=2)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_title('منحنی ROC', fontsize=14, fontweight='bold')
    ax4.set_xlabel('نرخ مثبت کاذب')
    ax4.set_ylabel('نرخ مثبت واقعی')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =============================================================================
# اجرای اصلی
# =============================================================================

def main():
    """تابع اصلی اجرای مدل"""
    
    print("🚀 شروع شبیه‌سازی مدل هوش آینده‌نگر...")
    print("=" * 60)
    
    # ایجاد سیستم
    system = AnticipatoryIntelligenceSystem()
    
    # تولید داده‌ها
    print("📊 در حال تولید داده‌های شبیه‌سازی شده...")
    data = system.simulate_data(n_steps=200)
    
    # اجرای مدل‌ها
    print("🔮 در حال اجرای مدل‌های پیش‌بینی...")
    
    results = {
        'traditional': [],
        'anticipatory': [],
        'actual': data['threat_level'].values
    }
    
    # اجرای مدل‌ها بر روی داده‌ها
    for i in range(50, len(data)):
        train_data = data.iloc[:i]
        
        # پیش‌بینی مدل سنتی
        trad_pred = system.traditional_model.predict_proba(train_data)
        results['traditional'].append(trad_pred)
        
        # پیش‌بینی مدل آینده‌نگر
        anti_pred = system.anticipatory_model.predict_proba(train_data)
        results['anticipatory'].append(anti_pred)
    
    # ارزیابی نتایج
    print("📈 در حال ارزیابی نتایج...")
    metrics = evaluate_models(results, data['threat_level'])
    
    # نمایش نتایج
    print("\n" + "=" * 60)
    print("نتایج مقایسه‌ای مدل آینده‌نگر vs مدل سنتی")
    print("=" * 60)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n📊 عملکرد مدل {model_name.upper()}:")
        print(f"   دقت (Precision): {model_metrics['precision']:.4f}")
        print(f"   recall (Recall): {model_metrics['recall']:.4f}")
        print(f"   امتیاز F1: {model_metrics['f1_score']:.4f}")
        print(f"   سطح زیر منحنی ROC: {model_metrics['auc_roc']:.4f}")
        print(f"   نرخ شناسایی زودهنگام: {model_metrics['early_detection_rate']:.4f}")
        print(f"   نرخ هشدار کاذب: {model_metrics['false_positive_rate']:.4f}")
    
    # تحلیل بهبودها
    print("\n" + "=" * 60)
    print("تحلیل بهبودهای مدل آینده‌نگر")
    print("=" * 60)
    
    trad_f1 = metrics['traditional']['f1_score']
    anti_f1 = metrics['anticipatory']['f1_score']
    improvement = (anti_f1 - trad_f1) / trad_f1 * 100
    
    early_improvement = (metrics['anticipatory']['early_detection_rate'] - 
                        metrics['traditional']['early_detection_rate']) * 100
    
    fp_reduction = (metrics['traditional']['false_positive_rate'] - 
                   metrics['anticipatory']['false_positive_rate']) * 100
    
    print(f"✅ بهبود عملکرد کلی (F1-Score): {improvement:+.1f}%")
    print(f"✅ بهبود شناسایی زودهنگام: {early_improvement:+.1f}%")
    print(f"✅ کاهش هشدارهای کاذب: {fp_reduction:+.1f}%")
    print(f"✅ بهبود قدرت تشخیص (AUC-ROC): {(metrics['anticipatory']['auc_roc'] - metrics['traditional']['auc_roc'])*100:+.1f}%")
    
    # ترسیم نمودارها
    print("\n🎨 در حال تولید نمودارها...")
    fig = plot_comparison(results, metrics, data)
    plt.show()
    
    # نمایش نمونه‌ای از پیش‌بینی‌ها
    print("\n" + "=" * 60)
    print("نمونه‌ای از پیش‌بینی‌های مدل‌ها")
    print("=" * 60)
    
    sample_size = min(10, len(results['traditional']))
    print(f"{'شماره':<8} {'واقعیت':<10} {'سنتی':<10} {'آینده‌نگر':<12} {'تفاوت':<10}")
    print("-" * 50)
    
    for i in range(sample_size):
        idx = -sample_size + i
        actual_val = results['actual'][idx]
        trad_val = results['traditional'][idx]
        anti_val = results['anticipatory'][idx]
        diff = anti_val - trad_val
        
        print(f"{i+1:<8} {actual_val:<10.3f} {trad_val:<10.3f} {anti_val:<12.3f} {diff:>+7.3f}")

if __name__ == "__main__":
    main()
