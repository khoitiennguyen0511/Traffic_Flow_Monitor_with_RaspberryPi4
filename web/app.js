// Cấu hình Firebase
const firebaseConfig = {
    apiKey: "AIzaSyDcQwwzi0I1bS_-U2Uf0l0HvOSLZVtuoxI",
    authDomain: "traffic-flow-monitor-e7ede.firebaseapp.com",
    databaseURL: "https://traffic-flow-monitor-e7ede-default-rtdb.firebaseio.com",
    projectId: "traffic-flow-monitor-e7ede",
    storageBucket: "traffic-flow-monitor-e7ede.firebasestorage.app",
    messagingSenderId: "139613832002",
    appId: "1:139613832002:web:b716917c7af5a6c411bedb",
    measurementId: "G-RQWZN4XV2V"
};

// Khởi tạo Firebase
firebase.initializeApp(firebaseConfig);
const database = firebase.database();
const dbRef = database.ref('traffic_system/latest_status');

// Map các trạng thái màu Firebase sang class CSS
const STATUS_CLASS_MAP = {
    'GREEN': 'green',
    'YELLOW': 'yellow',
    'RED': 'red',
    'ALL_OFF': 'off',
    'SLEEP': 'off'
};

/**
 * Cập nhật trạng thái màu cho đèn giao thông.
 * @param {string} lightId - ID của div đèn (ví dụ: 'light-1-status').
 * @param {string} state - Trạng thái từ Firebase (ví dụ: '1_GREEN', 'RED', 'ALL_OFF').
 */
function updateLightStatus(lightId, state) {
    const indicator = document.getElementById(lightId);
    indicator.className = 'light-indicator'; // Reset classes

    let color;

    // Phân tích màu sắc
    if (state.includes('_')) {
        // Lấy màu từ chuỗi Firebase (ví dụ: "1_GREEN" -> "GREEN")
        color = state.split('_')[1];
    } else {
        // Là màu đơn giản (ví dụ: "RED", "ALL_OFF")
        color = state;
    }

    const cssClass = STATUS_CLASS_MAP[color];
    
    if (cssClass) {
        indicator.classList.add(cssClass);
    } else {
        indicator.classList.add('off'); 
    }
}

/**
 * Hàm cập nhật Dashboard chính, lắng nghe thay đổi từ Firebase.
 */
dbRef.on('value', (snapshot) => {
    const data = snapshot.val();
    if (!data) return;

    const mode = data.esp32_mode || 'N/A';
    const density = data.latest_status || 'N/A';
    const currentState = data.esp32_state || 'ALL_OFF';

    // -----------------------------------
    // 1. Cập nhật Panel Tổng Quan
    // -----------------------------------
    
    // Cập nhật Mode và Density
    document.getElementById('mode-text').textContent = mode;
    document.getElementById('system-mode').className = `card mode-${mode.toLowerCase()}`;

    document.getElementById('density-text').textContent = density;
    document.getElementById('overall-status').className = `card status-${density.toLowerCase()}`;

    // Cập nhật Tổng Xe và Timestamp
    document.getElementById('total-vehicles').textContent = data.total_vehicles_all_time || 0;
    
    if (data.timestamp) {
        const date = new Date(data.timestamp * 1000);
        document.getElementById('timestamp').textContent = date.toLocaleTimeString('vi-VN');
    }

    // -----------------------------------
    // 2. Xử lý Thời Gian Đếm Ngược (ĐÃ CẬP NHẬT)
    // -----------------------------------
    
    const timeLeft1 = data.time_left_1 || 0;
    const timeLeft2 = data.time_left_2 || 0;
    
    let displayTime1;
    let displayTime2;
    
    // Kiểm tra các chế độ không sử dụng đếm ngược
    if (mode === 'MANUAL' || mode === 'SLEEP') { 
        // Hiển thị gạch ngang trong chế độ thủ công hoặc ngủ
        displayTime1 = '--';
        displayTime2 = '--';
    } else {
        // Nếu ở chế độ AUTO, hiển thị thời gian đếm ngược
        displayTime1 = `${timeLeft1.toString().padStart(2, '0')}s`;
        displayTime2 = `${timeLeft2.toString().padStart(2, '0')}s`;
    }
    
    document.getElementById('time-left-1').textContent = displayTime1;
    document.getElementById('time-left-2').textContent = displayTime2;


    // -----------------------------------
    // 3. Cập nhật Trạng Thái Đèn (Logic Đồng Bộ Hóa)
    // -----------------------------------
    
    let light1State = 'RED'; // Mặc định ĐỎ
    let light2State = 'RED'; // Mặc định ĐỎ

    if (currentState.startsWith('1_')) {
        // Làn 1 đang active (GREEN/YELLOW), Làn 2 phải ĐỎ
        light1State = currentState;
        // light2State giữ nguyên là 'RED'
    } else if (currentState.startsWith('2_')) {
        // Làn 2 đang active (GREEN/YELLOW), Làn 1 phải ĐỎ
        // light1State giữ nguyên là 'RED'
        light2State = currentState;
    } else if (currentState === 'ALL_OFF' || currentState === 'SLEEP') {
         // Chế độ Tắt/Ngủ
        light1State = currentState;
        light2State = currentState; 
    }
    
    updateLightStatus('light-1-status', light1State); 
    updateLightStatus('light-2-status', light2State); 

    // -----------------------------------
    // 4. Cập nhật Chi Tiết Vùng (Region Details)
    // -----------------------------------
    const regionDetailsDiv = document.getElementById('region-details');
    regionDetailsDiv.innerHTML = ''; 

    const regionCounts = data.region_counts;
    if (regionCounts) {
        let regionCardsHTML = '';
        
        Object.keys(regionCounts).forEach(key => {
            const region = regionCounts[key];
            
            regionCardsHTML += `
                <div class="region-card">
                    <h4>${key} (Tổng: ${region.total_in_region || 0})</h4>
                    <div class="vehicle-type">Motorbike: <span class="vehicle-count">${region.motorbike || 0}</span></div>
                    <div class="vehicle-type">Car: <span class="vehicle-count">${region.car || 0}</span></div>
                    <div class="vehicle-type">Bus: <span class="vehicle-count">${region.bus || 0}</span></div>
                    <div class="vehicle-type">Truck: <span class="vehicle-count">${region.truck || 0}</span></div>
                </div>
            `;
        });
        
        regionDetailsDiv.innerHTML = regionCardsHTML;
    }
});