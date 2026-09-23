from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp
import math

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

class MemeApp(App):
    def build(self):
        # Elemento de interface do Kivy para exibir o meme no ecrã
        self.img_widget = Image()

        # Configuração de dimensões padrão
        self.TAMANHO_PADRAO = (600, 600)

        # Mapeamento de imagens
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
            "cachorro_xd": "gatos/xd.jpg"
        }

        # Carregamento e redimensionamento das imagens
        self.imagens = {}
        for chave, caminho in imagens_paths.items():
            img = cv2.imread(caminho)
            if img is not None:
                self.imagens[chave] = cv2.resize(img, self.TAMANHO_PADRAO)
            else:
                print(f"Erro ao carregar: {caminho}")

        # Inicialização do MediaPipe e da Câmara
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6
        )

        # Agenda a execução do ciclo de processamento a 30 FPS
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img_widget

    def update(self, dt):
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_face = self.face_mesh.process(rgb_frame)
        results_hands = self.hands.process(rgb_frame)

        imagem_exibicao = self.imagens.get("gato_normal")
        acao_detectada = False

        # 1. DETEÇÃO DE MÃOS
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
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

                # GATO PENSANDO
                if results_face.multi_face_landmarks:
                    queixo_y = results_face.multi_face_landmarks[0].landmark[152].y
                    nariz_x = results_face.multi_face_landmarks[0].landmark[1].x
                    boca_y = results_face.multi_face_landmarks[0].landmark[13].y

                    mao_no_queixo = abs(indicador_ponta.y - queixo_y) < 0.08 or abs(indicador_ponta.y - boca_y) < 0.08
                    mao_centralizada = abs(indicador_ponta.x - nariz_x) < 0.12

                    if mao_no_queixo and mao_centralizada:
                        imagem_exibicao = self.imagens.get("gato_pensando")
                        acao_detectada = True

                # DEDO DO MEIO
                if not acao_detectada and (d_medio > d_medio_dobra) and \
                   (d_indicador < d_indicador_dobra) and \
                   (d_anelar < d_anelar_dobra) and \
                   (d_minimo < d_minimo_dobra):
                    imagem_exibicao = self.imagens.get("gato_meio")
                    acao_detectada = True

                # PEGAR PIRULITO
                elif not acao_detectada and dist(polegar_ponta, indicador_ponta) < 0.07:
                    imagem_exibicao = self.imagens.get("cachorro_pirulito")
                    acao_detectada = True

                # HANG LOOSE
                elif not acao_detectada and (d_polegar > d_polegar_dobra) and \
                     (d_minimo > d_minimo_dobra) and \
                     (d_indicador < d_indicador_dobra) and \
                     (d_medio < d_medio_dobra) and \
                     (d_anelar < d_anelar_dobra):
                    imagem_exibicao = self.imagens.get("gato_tranquilo")
                    acao_detectada = True

                # SOCO
                elif not acao_detectada and comprimento_mao < 0.12:
                    imagem_exibicao = self.imagens.get("gato_soco")
                    acao_detectada = True

                # INDICADOR LEVANTADO (Sabo)
                elif not acao_detectada and (d_indicador > d_indicador_dobra) and \
                     (d_medio < d_medio_dobra) and \
                     (d_anelar < d_anelar_dobra) and \
                     (d_minimo < d_minimo_dobra):
                    imagem_exibicao = self.imagens.get("gato_sabo")
                    acao_detectada = True

                # TERNO / GRAVATA
                elif not acao_detectada and results_face.multi_face_landmarks:
                    queixo_y = results_face.multi_face_landmarks[0].landmark[152].y
                    nariz_x = results_face.multi_face_landmarks[0].landmark[1].x

                    mao_no_centro = abs(pulso.x - nariz_x) < 0.08
                    mao_na_gravata = (queixo_y + 0.10) < pulso.y < (queixo_y + 0.30)

                    if mao_no_centro and mao_na_gravata:
                        imagem_exibicao = self.imagens.get("cachorro_terno")
                        acao_detectada = True

                # ÓCULOS
                if not acao_detectada and (indicador_ponta.y < pulso.y - 0.25 and medio_ponta.y < pulso.y - 0.25):
                    imagem_exibicao = self.imagens.get("gato_oculos")
                    acao_detectada = True

        # 2. DETEÇÃO FACIAL
        if not acao_detectada and results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                abertura_boca = face_landmarks.landmark[14].y - face_landmarks.landmark[13].y
                largura_rosto = dist(face_landmarks.landmark[234], face_landmarks.landmark[454])
                largura_boca = dist(face_landmarks.landmark[61], face_landmarks.landmark[291])
                proporcao_sorriso = largura_boca / (largura_rosto + 1e-6)
                olho_direito_alt = abs(face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)

                nariz_x = face_landmarks.landmark[1].x
                face_esq_x = face_landmarks.landmark[234].x
                face_dir_x = face_landmarks.landmark[454].x
                rotacao_cabeca = nariz_x - ((face_esq_x + face_dir_x) / 2)

                # BOCA MUITO ABERTA (Língua / Boca grande)
                if abertura_boca > 0.05:
                    imagem_exibicao = self.imagens.get("gato_boca")

                # BOCA UM POUCO ABERTA (Cachorro XD)
                elif 0.015 < abertura_boca <= 0.05:
                    imagem_exibicao = self.imagens.get("cachorro_xd")

                # SORRISO
                elif proporcao_sorriso > 0.43:
                    imagem_exibicao = self.imagens.get("cachorro_sorrindo")

                # CABEÇA DE LADO
                elif rotacao_cabeca > 0.04 or rotacao_cabeca < -0.04:
                    imagem_exibicao = self.imagens.get("gato_desconfiado")

                # OLHO ABERTO
                elif olho_direito_alt > 0.025:
                    imagem_exibicao = self.imagens.get("gato_olho_direito")

                else:
                    imagem_exibicao = self.imagens.get("gato_normal")

        # Converte a imagem para exibição na interface do Kivy
        if imagem_exibicao is not None:
            buffer = cv2.flip(imagem_exibicao, 0).tobytes()
            texture = Texture.create(
                size=(imagem_exibicao.shape[1], imagem_exibicao.shape[0]),
                colorfmt='bgr'
            )
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='unsigned byte')
            self.img_widget.texture = texture

    def on_stop(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    MemeApp().run()