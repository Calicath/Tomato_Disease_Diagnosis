import json
import os
import gradio as gr
import tensorflow as tf
from PIL import Image

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用 TensorFlow SavedModel
MODEL_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "models", "exported_model")
)

LABEL_MAP_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "models", "trained_model", "label_mapping.json")
)

# 加载 SavedModel
model = tf.keras.layers.TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

# 加载标签
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    label_mapping = json.load(f)

# 加载防治建议
advice_mapping = {
    "Tomato_Bacterial_spot": {
        "title": "细菌性斑点病专业防控方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">🔥</span> 立即摘除病叶病株焚烧，周边3米内植株喷洒1%过氧乙酸消毒</li>
                <li><span class="icon">🧪</span> 铜制剂方案：20%噻菌铜SC 800倍液 + 3%中生菌素WP 1000倍液，间隔5天连喷3次</li>
                <li><span class="icon">💧</span> 水肥管理：EC值控制在1.8-2.2 mS/cm，滴灌每次≤2小时</li>
                <li><span class="icon">🔄</span> 轮作制度：玉米→洋葱→番茄（三年周期）</li>
                <li><span class="icon">🌡️</span> 种子处理：55℃温汤浸种30分钟，1%次氯酸钠浸泡10分钟</li>
            </ul>
            <div class="warning"><i class="fas fa-exclamation-triangle"></i> 避免与茄科作物邻作，隔离带≥50米</div>
        </div>
        """
    },
    "Tomato_Leaf_Mold": {
        "title": "叶霉病综合防控方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 治疗剂：40%氟硅唑EC 4000倍液 + 25%吡唑醚菌酯SC 1500倍液</li>
                <li><span class="icon">🌡️</span> 环境控制：昼温28±2℃，夜温≥15℃，相对湿度≤75%</li>
                <li><span class="icon">✂️</span> 整枝规范：保留顶部6片叶，摘除距地面40cm以下叶片</li>
                <li><span class="icon">🧼</span> 设施消毒：10%腐霉利烟剂500g/亩熏棚，密闭48小时</li>
                <li><span class="icon">📏</span> 密植标准：大果型≤1800株/亩，樱桃番茄≤2200株/亩</li>
            </ul>
            <div class="pro-tip"><i class="fas fa-lightbulb"></i> 推荐PO膜透光率＞85%，晨间通风1小时除湿</div>
        </div>
        """
    },
    "Tomato_Early_blight": {
        "title": "早疫病精准防治方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 70%丙森锌WP 600倍液 + 28%烯肟菌酯SC 2000倍液，叶片正反喷雾</li>
                <li><span class="icon">⚖️</span> 营养调控：叶面喷施0.3%磷酸二氢钾 + 0.1%芸苔素内酯</li>
                <li><span class="icon">🌧️</span> 雨季管理：开挖排水沟（深30cm，坡降3%），覆盖黑色地膜</li>
                <li><span class="icon">🔄</span> 轮作模式：番茄→水稻→十字花科（两年周期）</li>
                <li><span class="icon">🔍</span> 监测标准：每周调查病叶率，超过5%立即用药</li>
            </ul>
        </div>
        """
    },
    "Tomato_Late_blight": {
        "title": "晚疫病紧急处置方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 特效药：68.75%氟吡菌胺·烯酰吗啉SC 1000倍液茎基部灌注</li>
                <li><span class="icon">🌡️</span> 环境调控：维持夜间温度＞15℃，降低结露时间≤4小时/天</li>
                <li><span class="icon">🛡️</span> 物理阻隔：行间铺设稻壳（厚度≥5cm）防止病菌飞溅</li>
                <li><span class="icon">🔥</span> 病株处理：10%硫酸铜溶液灌穴，覆土密封</li>
                <li><span class="icon">📅</span> 用药周期：首诊后第1、3、7天三次用药</li>
            </ul>
            <div class="warning"><i class="fas fa-exclamation-triangle"></i> 禁止大水漫灌，雨后立即喷施保护剂</div>
        </div>
        """
    },
    "Tomato_Septoria_leaf_spot": {
        "title": "壳针孢叶斑病防控方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 50%腐霉利WP 800倍液 + 75%百菌清WP 600倍液交替使用</li>
                <li><span class="icon">✂️</span> 清除病残体：摘除植株下部20cm内叶片并深埋</li>
                <li><span class="icon">💧</span> 湿度控制：清晨通风1小时，使相对湿度≤80%</li>
                <li><span class="icon">🌱</span> 抗性品种：选用'金棚1号'、'中杂9号'等抗病品种</li>
                <li><span class="icon">🧪</span> 土壤处理：定植前亩施50%克菌丹可湿性粉剂3kg</li>
            </ul>
        </div>
        """
    },
    "Tomato_Target_Spot": {
        "title": "靶斑病综合防治方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 43%氟菌·肟菌酯SC 3000倍液 + 50%异菌脲WP 1000倍液</li>
                <li><span class="icon">🌡️</span> 温度管理：维持昼温25-28℃，夜温＞15℃减少结露</li>
                <li><span class="icon">✂️</span> 农事操作：避免中午高温时段整枝打杈</li>
                <li><span class="icon">🔄</span> 轮作制度：与禾本科作物轮作2年以上</li>
                <li><span class="icon">🔍</span> 监测预警：悬挂孢子捕捉器，密度＞5个/视野时预警</li>
            </ul>
            <div class="pro-tip"><i class="fas fa-lightbulb"></i> 喷药后覆盖无纺布提高药效</div>
        </div>
        """
    },
    "Tomato_mosaic_virus": {
        "title": "花叶病毒病紧急处置方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 病毒抑制剂：8%宁南霉素AS 1000倍液 + 0.004%芸苔素内酯</li>
                <li><span class="icon">🛡️</span> 虫媒防控：10%吡虫啉WP 2000倍液灭蚜，设置黄色粘虫板</li>
                <li><span class="icon">🔥</span> 病株处理：发现即拔除，病穴撒生石灰消毒</li>
                <li><span class="icon">🌱</span> 抗病品种：选用'浙粉202'、'金冠5号'等品种</li>
                <li><span class="icon">🧼</span> 器械消毒：工具用10%磷酸三钠浸泡10分钟</li>
            </ul>
            <div class="warning"><i class="fas fa-exclamation-triangle"></i> 严禁吸烟操作，烟草花叶病毒易通过接触传播</div>
        </div>
        """
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "黄化曲叶病毒病系统防控",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">🦟</span> 粉虱防控：22%螺虫乙酯SC 1500倍液 + 悬挂黄色粘虫板（30块/亩）</li>
                <li><span class="icon">🌐</span> 隔离防护：40目防虫网全程覆盖，进出口设双层缓冲间</li>
                <li><span class="icon">🔥</span> 高温闷棚：7-8月密闭大棚，地表温度达70℃持续15天</li>
                <li><span class="icon">🌱</span> 抗病品种：'浙粉702'、'金棚8号'等高抗品种</li>
                <li><span class="icon">📅</span> 种植周期：避开发病高峰，选择春提早或秋延后栽培</li>
            </ul>
        </div>
        """
    },
    "Tomato_Spider_mites": {
        "title": "二斑叶螨综合防控方案",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">💊</span> 杀螨剂：24%螺螨酯SC 3000倍液 + 5%阿维菌素EC 1500倍液</li>
                <li><span class="icon">🦠</span> 生物防治：释放加州新小绥螨（2000头/亩，每15天1次）</li>
                <li><span class="icon">💧</span> 环境调控：叶背喷水（每天10:00-11:00），保持湿度＞60%</li>
                <li><span class="icon">🧹</span> 清园处理：及时清除田间杂草及残株</li>
                <li><span class="icon">🔄</span> 轮作制度：与水稻轮作，水旱交替破坏螨虫生境</li>
            </ul>
            <div class="pro-tip"><i class="fas fa-lightbulb"></i> 重点喷施叶背，加入有机硅助剂提高附着</div>
        </div>
        """
    },
    "healthy": {
        "title": "健康植株养护建议",
        "content": """
        <div class="expert-advice">
            <ul>
                <li><span class="icon">📊</span> 环境监控：昼温25-28℃/夜温15-18℃，光照≥6小时/天</li>
                <li><span class="icon">⚖️</span> 水肥方案：EC 1.8-2.2 mS/cm，N-P-K 1:0.5:2.5，每7天检测1次</li>
                <li><span class="icon">🔬</span> 叶片诊断：SPAD值维持45-55，叶绿素荧光参数Fv/Fm≥0.8</li>
                <li><span class="icon">🛡️</span> 预防方案：0.3%氨基寡糖素AS 800倍液，每15天喷施</li>
                <li><span class="icon">📈</span> 生长监测：株高日增长量1.5-2cm，茎粗≥8mm（开花期）</li>
            </ul>
            <div class="pro-tip"><i class="fas fa-lightbulb"></i> 建议安装物联网传感器实时监控环境参数</div>
        </div>
        """
    }
}

# 图像预处理
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array)

# 预测函数
def predict(image):
    try:
        if not isinstance(image, Image.Image):
            return None, "输入类型错误", "<div class='error'>请上传有效的图像文件</div>"

        # 执行预测
        img_array = preprocess_image(image)

        # TFSMLayer 调用 SavedModel
        output = model(img_array)

        # SavedModel 的输出名称是 dense_1
        pred = output["dense_1"].numpy()

        # 获取前三个预测结果
        sorted_results = sorted(
            [(label_mapping[str(i)], float(pred[0][i]))
            for i in range(len(label_mapping))],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # 生成预览图
        preview_img = image.copy().resize((300, 300))

        # 生成置信度显示
        confidence = f"诊断置信度: {sorted_results[0][1] * 100:.1f}%"

        # 生成HTML诊断结果
        diagnosis_html = "<div class='diagnosis-results'>"

        # 疾病概率显示
        for name, prob in sorted_results:
            status_class = "disease"
            diagnosis_html += f"""
            <div class='diagnosis-item {status_class}'>
                <div class='disease-name'>{name}</div>
                <div class='probability-bar'>
                    <div style='width:{prob * 100}%'></div>
                </div>
                <div class='probability-value'>{prob * 100:.1f}%</div>
            </div>
            """

        # 疾病与建议映射关系
        disease_mapping = {
            "番茄细菌性斑点病": "Tomato_Bacterial_spot",
            "健康": "healthy",
            "番茄早疫病": "Tomato_Early_blight",
            "番茄晚疫病": "Tomato_Late_blight",
            "番茄叶霉病": "Tomato_Leaf_Mold",
            "番茄壳针孢叶斑病": "Tomato_Septoria_leaf_spot",
            "番茄二斑叶螨": "Tomato_Spider_mites",
            "番茄靶斑病": "Tomato_Target_Spot",
            "番茄花叶病毒病": "Tomato_mosaic_virus",
            "番茄黄化曲叶病毒病": "Tomato_Yellow_Leaf_Curl_Virus"
        }

        # 获取主病害建议
        main_disease = sorted_results[0][0]
        if main_disease != "00":
            # 清理键名中的空格
            clean_disease = main_disease.strip()
            advice_key = disease_mapping.get(clean_disease, "healthy")
            advice = advice_mapping[advice_key]

            # 添加病害建议
            diagnosis_html += f"""
            <div class="advice-container active">
                <div class="advice-header disease-header">
                    <i class="fas fa-bug"></i>
                    {advice['title']}
                </div>
                <div class="advice-content">
                    {advice['content']}
                    <div class="footer-note">* </div>
                </div>
            </div>
            """

        diagnosis_html += "</div>"
        return preview_img, confidence, diagnosis_html

    except Exception as e:
        return None, "系统错误", f"<div class='error'>{str(e)}</div>"

# CSS增强
professional_css = """
:root {
    --primary: #2b8ec8;
    --success: #4CAF50;
    --danger: #f44336;
}
#main-container {
    display: grid;
    grid-template-columns: 1fr 1.2fr; /* 调整左右比例 */
    gap: 32px;
    max-width: 1400px !important;
}

.card {
    background: white;
    border-radius: 12px !important;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 24px;
    border: none !important;
}
.card-title {
    font: 600 18px/1.2 'Segoe UI', sans-serif;
    color: var(--text);
    margin-bottom: 16px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
#upload-box {
    border: 2px dashed var(--primary) !important;
    border-radius: 12px;
    transition: all 0.3s ease;
    min-height: 400px;
}
#upload-box:hover {
    border-color: var(--primary) !important;
    background: var(--secondary) !important;
}
.diagnosis-item {
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 16px;
}

.disease-name {
    color: #d32f2f !important;
    font-weight: 600 !important;
}
.disease {
    background: #fce4ec !important;
    border-left: 4px solid var(--danger) !important;
}
.healthy {
    background: #f1f8e9 !important;
    border-left: 4px solid var(--success) !important;
}
.probability-bar {
    flex: 1;
    height: 12px;
    background: #eee;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}
.probability-bar div {
    height: 100%;
    background: linear-gradient(90deg, var(--primary) 0%, #1e5f8d 100%);
    transition: width 0.5s ease;
}
.preview-image {
    border-radius: 12px;
    border: 3px solid white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.confidence-badge {
    background: var(--primary);
    color: white !important;
    padding: 8px 16px;
    border-radius: 20px;
    font: 500 14px/1 'Segoe UI';
    display: inline-block;
}
.examples .thumbnail {
    border: 2px solid transparent;
    transition: all 0.3s ease;
    border-radius: 8px;
    cursor: pointer;
}
.examples .thumbnail:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(43,142,200,0.2);
    border-color: var(--primary);
}
.loading-overlay {
    background: rgba(255,255,255,0.9) !important;
    border-radius: 12px !important;
}
.advice-section {
    background: #fff3e0;
    border-radius: 12px;
    margin-top: 24px;
    padding: 20px;
    border-left: 4px solid #ffa726;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.advice-title {
    color: #d84315 !important;
    font: 600 18px/1.5 'Segoe UI';
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.advice-title::before {
    content: "⚠️";
    margin-right: 8px;
}

.advice-content ul {
    margin: 12px 0;
    padding-left: 24px;
}

.advice-content li {
    margin: 8px 0;
    line-height: 1.6;
    color: #5d4037;
}

.advice-content li::marker {
    color: #ffa726;
}
.expert-advice {
    background: #fff9f2;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    border-left: 4px solid #ff9800;
}

.expert-advice h4 {
    color: #d32f2f;
    font-size: 16px;
    margin: 0 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #eee;
}

.expert-advice ul {
    list-style: none;
    padding-left: 0;
}

.expert-advice li {
    margin: 12px 0;
    padding: 10px;
    background: #fff;
    border-radius: 6px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    display: flex;
    align-items: center;
    gap: 10px;
}

.expert-advice .icon {
    background: #ff9800;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.warning {
    background: #ffebee;
    color: #c62828;
    padding: 12px;
    border-radius: 8px;
    margin-top: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.pro-tip {
    background: #e3f2fd;
    color: #1976d2;
    padding: 12px;
    border-radius: 8px;
    margin-top: 15px;
}

.fas {
    margin-right: 8px;
}
.advice-container {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-top: 20px;
    transition: all 0.3s ease;
}

.advice-header {
    padding: 16px 24px;
    background: linear-gradient(135deg, #ffa726 0%, #f57c00 100%);
    color: white !important;
    border-radius: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    font: 600 16px/1.5 'Segoe UI';
}

.advice-header .fa-chevron-down {
    transition: transform 0.3s ease;
    font-size: 14px;
}

.advice-content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease-out;
    padding: 0 24px;
}

.advice-container.active .advice-content {
    max-height: 1000px;
    padding: 24px;
}

.advice-container.active .fa-chevron-down {
    transform: rotate(180deg);
}

.footer-note {
    margin-top: 20px;
    padding-top: 12px;
    border-top: 1px dashed #ddd;
    color: #666;
    font-size: 12px;
    text-align: right;
}

.expert-advice li {
    margin: 10px 0;
    padding: 12px;
    gap: 12px;
}

.expert-advice .icon {
    width: 28px;
    height: 28px;
}

.pro-tip, .warning {
    margin: 15px 0;
    padding: 15px;
}
.advice-container:last-child {
    display: block !important;
    opacity: 1 !important;
}

.diagnosis-item.healthy {
    display: none !important;
}
.upload-column {
    min-width: 480px !important;
}

.diagnosis-column {
    min-width: 600px !important;
}

/* 病害建议样式 */
.disease-header {
    background: linear-gradient(135deg, #ff6666 0%, #cc0000 100%) !important;
}

/* 健康建议样式 */
.health-advice {
    border: 2px solid #4CAF50;
    background: #f8fff8 !important;
}
.health-header {
    background: linear-gradient(135deg, #66bb6a 0%, #2e7d32 100%) !important;
}

/* 响应式布局 */
@media (max-width: 1200px) {
    #main-container {
        grid-template-columns: 1fr;
    }
}
"""
js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // 自动展开健康建议
    const healthAdvice = document.querySelector('.advice-container:last-child');
    if(healthAdvice) {
        healthAdvice.classList.add('active');
        healthAdvice.querySelector('.advice-content').style.maxHeight = '1000px';
    }
    
    // 病害建议折叠功能
    document.body.addEventListener('click', function(e) {
        const header = e.target.closest('.advice-header:not(:last-child)');
        if (header) {
            const container = header.parentElement;
            container.classList.toggle('active');
        }
    });
});
</script>
 """
# 加载Gradio应用
with gr.Blocks(title="番茄病害诊断系统", css=professional_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(js)

    with gr.Row(elem_id="main-container"):
        # 输入面板
        with gr.Column(scale=6, elem_classes="upload-column"):
            with gr.Group(elem_classes="card"):
                gr.Markdown("""<div classss=="card-title"><i cla"fas fa-upload"></i> 图像上传区</div>""")
                upload_box = gr.Image(
                    type="pil",
                    label="点击上传叶片图像",
                    elem_id="upload-box",
                    height=420,
                    image_mode="RGB"
                )
                with gr.Row():
                    gr.Examples(
                        examples=[
                            os.path.join(BASE_DIR, "examples", f)
                            for f in ["healthy.jpg", "disease1.jpg", "disease2.jpg"]
                        ],
                        inputs=[upload_box],
                        label="典型病例示例",
                        examples_per_page=3
                    )

        # 诊断面板
        with gr.Column(scale=6, elem_classes="diagnosis-column"):
            with gr.Group(elem_classes="card"):
                gr.Markdown("""<div class="card-title"><i class="fas fa-diagnoses"></i> 诊断报告</div>""")
                with gr.Column():
                    preview = gr.Image(
                        label="图像预览",
                        interactive=False,
                        height=240,
                        elem_classes="preview-image"
                    )
                    confidence = gr.Label(
                        label="置信度",
                        value="等待分析...",
                        elem_classes="confidence-badge",
                        show_label=False
                    )
                    diagnosis = gr.HTML("""
                    <div style='padding:32px; text-align:center; color:#888;'>
                        <i class="fas fa-microscope"></i> 上传叶片图像后自动生成诊断报告
                    </div>
                    """)



    # 交互逻辑
    upload_box.change(
        fn=predict,
        inputs=upload_box,
        outputs=[preview, confidence, diagnosis],
        api_name="diagnose",
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="localhost", # 仅在本地访问，可修改为0.0.0.0，允许外部访问
        server_port=7860, # 可修改为其他端口
        show_error=True,
        share=False
    )