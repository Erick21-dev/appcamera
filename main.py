import cv2
import mediapipe as mp
import math

# Função para calcular distância entre dois pontos
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ==========================================
# CONFIGURAÇÃO DE IMAGENS & PADRONIZAÇÃO
# ==========================================
# Define um tamanho fixo (Largura, Altura) para TODAS as janelas de memes
TAMANHO_PADRAO = (600, 600) 

imagens_paths = {
    "gato_normal": "gatos/gato.jpg",
    "gato_boca": "gatos/2462974793378144.jpg",
    "gato_olho_direito": "gatos/20+ Silly Kitties From Weird Angles That You and___.jpg",
    "gato_soco": "gatos/cute cat emojis and reaction pics.jpg",
    "gato_desconfiado": "gatos/giga cat.jpg",
    "gato_sabo": "gatos/43769427624677180.jpg",
    "gato_tranquilo": "gatos/49961877112718676.jpg",
    "cachorro_sorrindo": "gatos/cachorro sorrindo.jpg",
    "cachorro_terno": "gatos/terno.jpg",
    "gato_oculos": "gatos/gato oculos.jpg",
    "gato_pensando": "gatos/pensando.jpg",
    "cachorro_pirulito": "gatos/pirulito.jpg",
    "gato_meio": "gatos/meio.jpg",
    "cachorro_xd": "gatos/xd.jpg"  # <--- Imagem do cachorro encarando
}

imagens = {}
erro = False

# Carrega e REDIMENSIONA todas as imagens para o mesmo tamanho
for chave, caminho in imagens_paths.items():
    img = cv2.imread(caminho)
    if img is None:
        print(f"❌ Erro ao carregar: '{caminho}' (Verifique o nome/extensão do arquivo)")
        erro = True
    else:
        # Força todas as imagens a terem exatamente 600x600
        imagens[chave] = cv2.resize(img, TAMANHO_PADRAO)

if erro:
    print("\nO programa foi encerrado devido a falhas no carregamento de imagens.")
    exit()

# Atribuição das imagens
gato_normal = imagens["gato_normal"]
gato_boca = imagens["gato_boca"]
gato_olho_direito = imagens["gato_olho_direito"]
gato_soco = imagens["gato_soco"]
gato_desconfiado = imagens["gato_desconfiado"]
gato_sabo = imagens["gato_sabo"]
gato_tranquilo = imagens["gato_tranquilo"]
cachorro_sorrindo = imagens["cachorro_sorrindo"]
cachorro_terno = imagens["cachorro_terno"]
gato_oculos = imagens["gato_oculos"]
gato_meio = imagens["gato_meio"]
gato_pensando = imagens["gato_pensando"]
cachorro_pirulito = imagens["cachorro_pirulito"]
cachorro_xd = imagens["cachorro_xd"]

# ==========================================
# MEDIAPIPE & CÂMERA
# ==========================================
CAMERA = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAMERA)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    imagem_exibicao = gato_normal
    acao_detectada = False

    # 1. DETECÇÃO DE MÃOS
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            pulso = hand_landmarks.landmark[0]
            
            polegar_ponta = hand_landmarks.landmark[4]
            indicador_ponta = hand_landmarks.landmark[8]
            medio_ponta = hand_landmarks.landmark[12]
            anelar_ponta = hand_landmarks.landmark[16]
            minimo_ponta = hand_landmarks.landmark[20]
            
            indicador_dobra = hand_landmarks.landmark[6]
            medio_dobra = hand_landmarks.landmark[10]
            anelar_dobra = hand_landmarks.landmark[14]
            minimo_dobra = hand_landmarks.landmark[18]
            polegar_dobra = hand_landmarks.landmark[2]

            d_polegar = dist(polegar_ponta, pulso)
            d_indicador = dist(indicador_ponta, pulso)
            d_medio = dist(medio_ponta, pulso)
            d_anelar = dist(anelar_ponta, pulso)
            d_minimo = dist(minimo_ponta, pulso)

            d_polegar_dobra = dist(polegar_dobra, pulso)
            d_indicador_dobra = dist(indicador_dobra, pulso)
            d_medio_dobra = dist(medio_dobra, pulso)
            d_anelar_dobra = dist(anelar_dobra, pulso)
            d_minimo_dobra = dist(minimo_dobra, pulso)

            comprimento_mao = abs(medio_ponta.y - pulso.y)

            # GESTO: GATO PENSANDO (Mão/Indicador perto do queixo ou boca)
            if results_face.multi_face_landmarks:
                queixo_y = results_face.multi_face_landmarks[0].landmark[152].y
                nariz_x = results_face.multi_face_landmarks[0].landmark[1].x
                boca_y = results_face.multi_face_landmarks[0].landmark[13].y
                
                mao_no_queixo = abs(indicador_ponta.y - queixo_y) < 0.08 or abs(indicador_ponta.y - boca_y) < 0.08
                mao_centralizada = abs(indicador_ponta.x - nariz_x) < 0.12
                
                if mao_no_queixo and mao_centralizada:
                    imagem_exibicao = gato_pensando
                    acao_detectada = True

            # GESTO: DEDO DO MEIO
            if not acao_detectada and (d_medio > d_medio_dobra) and \
               (d_indicador < d_indicador_dobra) and \
               (d_anelar < d_anelar_dobra) and \
               (d_minimo < d_minimo_dobra):
                imagem_exibicao = gato_meio
                acao_detectada = True

            # GESTO: PEGAR PIRULITO
            elif not acao_detectada and dist(polegar_ponta, indicador_ponta) < 0.07:
                imagem_exibicao = cachorro_pirulito
                acao_detectada = True

            # GESTO: HANG LOOSE
            elif not acao_detectada and (d_polegar > d_polegar_dobra) and \
                 (d_minimo > d_minimo_dobra) and \
                 (d_indicador < d_indicador_dobra) and \
                 (d_medio < d_medio_dobra) and \
                 (d_anelar < d_anelar_dobra):
                imagem_exibicao = gato_tranquilo
                acao_detectada = True

            # GESTO: SOCO
            elif not acao_detectada and comprimento_mao < 0.12:
                imagem_exibicao = gato_soco
                acao_detectada = True

            # GESTO: INDICADOR LEVANTADO (Sabo)
            elif not acao_detectada and (d_indicador > d_indicador_dobra) and \
                 (d_medio < d_medio_dobra) and \
                 (d_anelar < d_anelar_dobra) and \
                 (d_minimo < d_minimo_dobra):
                imagem_exibicao = gato_sabo
                acao_detectada = True

            # GESTO: GRAVATA / TERNO
            elif not acao_detectada and results_face.multi_face_landmarks:
                queixo_y = results_face.multi_face_landmarks[0].landmark[152].y
                nariz_x = results_face.multi_face_landmarks[0].landmark[1].x
                
                mao_no_centro = abs(pulso.x - nariz_x) < 0.08
                mao_na_gravata = (queixo_y + 0.10) < pulso.y < (queixo_y + 0.30)
                
                if mao_no_centro and mao_na_gravata:
                    imagem_exibicao = cachorro_terno
                    acao_detectada = True

            # GESTO: DOIS DEDOS (Óculos)
            if not acao_detectada and (indicador_ponta.y < pulso.y - 0.25 and medio_ponta.y < pulso.y - 0.25):
                imagem_exibicao = gato_oculos
                acao_detectada = True

    # 2. DETECÇÃO FACIAL
    if not acao_detectada and results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            
            # Mede a distância vertical entre os lábios internos
            abertura_boca = face_landmarks.landmark[14].y - face_landmarks.landmark[13].y
            
            # Cálculo de Sorriso proporcional à largura do rosto
            largura_rosto = dist(face_landmarks.landmark[234], face_landmarks.landmark[454])
            largura_boca = dist(face_landmarks.landmark[61], face_landmarks.landmark[291])
            proporcao_sorriso = largura_boca / (largura_rosto + 1e-6)

            olho_direito_alt = abs(face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
            
            nariz_x = face_landmarks.landmark[1].x
            face_esq_x = face_landmarks.landmark[234].x
            face_dir_x = face_landmarks.landmark[454].x
            rotacao_cabeca = nariz_x - ((face_esq_x + face_dir_x) / 2)

            # 1. Boca MUITO aberta -> Gato com boca aberta / Mostrar língua (> 0.05)
            if abertura_boca > 0.05:
                imagem_exibicao = gato_boca

            # 2. Boca LEVEMENTE aberta -> Cachorro XD (entre 0.015 e 0.05)
            elif 0.015 < abertura_boca <= 0.05:
                imagem_exibicao = cachorro_xd

            # 3. Sorriso
            elif proporcao_sorriso > 0.43:
                imagem_exibicao = cachorro_sorrindo

            # 4. Olhar de lado / Cabeça virada
            elif rotacao_cabeca > 0.04 or rotacao_cabeca < -0.04:  
                imagem_exibicao = gato_desconfiado

            # 5. Olho aberto/piscada
            elif olho_direito_alt > 0.025:
                imagem_exibicao = gato_olho_direito

            else:
                imagem_exibicao = gato_normal

    # Exibição das janelas
    cv2.imshow("Sua Camera", frame)
    cv2.imshow("Gato Meme", imagem_exibicao)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()