import streamlit as st
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(page_title="友商技术名信息更新", layout="wide")

# 创建示例数据
def create_sample_competitor_data():
    """创建竞争对手产品数据示例"""
    data = {
        'index': range(1, 6),
        'competitor_product': [f'Product_{i}' for i in range(1, 6)],
        'competitor_name': [f'Company_{i}' for i in range(1, 6)],
        'uih_product': [f'UIH_Product_{i}' for i in range(1, 6)],
        'price': np.random.randint(1000, 10000, 5),
        'performance': np.random.rand(5) * 100
    }
    return pd.DataFrame(data)

def create_sample_uih_data():
    """创建UIH产品数据示例"""
    data = {
        'index': range(1, 6),
        'competitor_product': [f'Competitor_{i}' for i in range(1, 6)],
        'competitor_name': [f'Brand_{i}' for i in range(1, 6)],
        'uih_product': [f'UIH_Solution_{i}' for i in range(1, 6)],
        'market_share': np.random.rand(5) * 50,
        'rating': np.random.rand(5) * 5
    }
    return pd.DataFrame(data)

def main():
    # 添加自定义CSS样式以优化对齐
    st.markdown("""
    <style>
    /* 优化选择框和表格的垂直对齐 */
    .stSelectbox {
        margin-top: 0px !important;
    }
    
    .stSelectbox > div > div {
        margin-top: 0px !important;
    }
    
    /* 确保数据表格和选择框在同一水平线上 */
    .stDataFrame {
        margin-top: 0px !important;
    }
    
    /* 优化容器间距 */
    .block-container {
        padding-top: 1rem;
    }
    
    /* 统一标题样式 */
    .stMarkdown h3 {
        margin-bottom: 0.5rem !important;
    }
    
    /* 优化markdown在表格中间的显示效果 */
    .table-center-content {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .table-center-content strong {
        color: #495057;
        font-size: 14px;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🔧 Step 2: 更新友商技术名信息")
    st.markdown("根据Step1的结果，选择恰当的竞争对手产品和UIH产品信息")
    
    # 操作说明
    with st.expander("💡 操作说明"):
        st.markdown("""
        完成Step1后，在下方表格中查看提取的产品信息，然后使用右侧的下拉菜单选择恰当的产品和索引
        """)
    
    # Step 2.1: 竞争对手产品信息
    st.markdown("## 🏢 Step 2.1: 竞争对手产品信息")
    st.markdown("### competitor_product_info 数据表")
    st.markdown("竞争对手产品数据")
    
    # 创建竞争对手数据
    competitor_df = create_sample_competitor_data()
    
    # 使用容器和列布局确保对齐
    container1 = st.container()
    with container1:
        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        
        with col1:
            # 数据表格
            st.dataframe(
                competitor_df,
                use_container_width=True,
                height=200,
                hide_index=True
            )
        
        with col2:
            # 使用HTML和CSS来精确控制markdown在表格中间的位置
            st.markdown("""
            <div style="display: flex; flex-direction: column; height: 200px; justify-content: center; align-items: center;">
                <div class="table-center-content">
                    <div style="margin-bottom: 10px;">
                        <strong>判断在语义上友商技术是否属于UIH</strong>
                    </div>
                    <div>
                        <strong>选择数据行索引</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 选择框放在markdown下方
            competitor_product_options = competitor_df['competitor_product'].tolist()
            selected_competitor_product = st.selectbox(
                "选择 competitor_product",
                options=competitor_product_options,
                key="competitor_product_select",
                label_visibility="collapsed"
            )
        
        with col3:
            # 使用HTML和CSS来精确控制markdown在表格中间的位置
            st.markdown("""
            <div style="display: flex; flex-direction: column; height: 200px; justify-content: center; align-items: center;">
                <div class="table-center-content">
                    <div style="margin-bottom: 10px;">
                        <strong>判断在语义上UIH技术属于哪个候选项</strong>
                    </div>
                    <div>
                        <strong>选择数据行索引</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 选择框放在markdown下方
            competitor_index_options = competitor_df['index'].tolist()
            selected_competitor_index = st.selectbox(
                "选择数据行索引",
                options=competitor_index_options,
                key="competitor_index_select",
                label_visibility="collapsed"
            )
    
    # 添加分隔线
    st.divider()
    
    # Step 2.2: UIH产品信息
    st.markdown("## 🏥 Step 2.2: UIH产品信息")
    st.markdown("### uih_product_info 数据表")
    st.markdown("UIH产品数据")
    
    # 创建UIH数据
    uih_df = create_sample_uih_data()
    
    # 使用容器和列布局确保对齐
    container2 = st.container()
    with container2:
        col1, col2, col3 = st.columns([3, 1.5, 1.5])
        
        with col1:
            # 数据表格
            st.dataframe(
                uih_df,
                use_container_width=True,
                height=200,
                hide_index=True
            )
        
        with col2:
            # 使用HTML和CSS来精确控制markdown在表格中间的位置
            st.markdown("""
            <div style="display: flex; flex-direction: column; height: 200px; justify-content: center; align-items: center;">
                <div class="table-center-content">
                    <div style="margin-bottom: 10px;">
                        <strong>判断在语义上友商技术是否属于UIH</strong>
                    </div>
                    <div>
                        <strong>选择数据行索引</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 选择框放在markdown下方
            uih_product_options = uih_df['uih_product'].tolist()
            selected_uih_product = st.selectbox(
                "选择 uih_product",
                options=uih_product_options,
                key="uih_product_select",
                label_visibility="collapsed"
            )
        
        with col3:
            # 使用HTML和CSS来精确控制markdown在表格中间的位置
            st.markdown("""
            <div style="display: flex; flex-direction: column; height: 200px; justify-content: center; align-items: center;">
                <div class="table-center-content">
                    <div style="margin-bottom: 10px;">
                        <strong>判断在语义上UIH技术属于哪个候选项</strong>
                    </div>
                    <div>
                        <strong>选择数据行索引</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 选择框放在markdown下方
            uih_index_options = uih_df['index'].tolist()
            selected_uih_index = st.selectbox(
                "选择数据行索引",
                options=uih_index_options,
                key="uih_index_select",
                label_visibility="collapsed"
            )
    
    # 显示选择结果
    st.markdown("---")
    st.markdown("### 📋 选择结果")
    
    results_col1, results_col2 = st.columns(2)
    
    with results_col1:
        st.markdown("**竞争对手产品选择:**")
        st.info(f"产品: {selected_competitor_product} (索引: {selected_competitor_index})")
    
    with results_col2:
        st.markdown("**UIH产品选择:**")
        st.info(f"产品: {selected_uih_product} (索引: {selected_uih_index})")
    
    # 添加确认按钮
    if st.button("✅ 确认选择", type="primary", use_container_width=True):
        st.success("✅ 选择已确认！")
        st.balloons()

if __name__ == "__main__":
    main()