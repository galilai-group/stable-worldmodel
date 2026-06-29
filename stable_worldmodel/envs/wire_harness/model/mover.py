"""
MOVER - Roboter/Agent-Klasse aus dem Original-Skript
"""

import math
import numpy as np
import heapq


class Mover:
    """
    ORIGINAL-KLASSE aus Environment (innere Klasse).
    Repräsentiert einen einzelnen Roboter/Mover in der Simulation.
    
    Was ein Mover macht:
    - Verwaltet eigene Position und Ziel
    - Führt lokale Kollisionserkennung durch
    - Wählt Aktionen basierend auf Constraints
    - Plant Pfade mit A* oder direkter Bewegung
    """
    
    def __init__(self, env, mu_index, mu_start, mu_joint, mu_start_move, 
                 follow, max_dist, vel, cable_connect, cable_start_mu):
        """
        Initialisiert einen Mover.
        
        Was hier eingerichtet wird:
        - Positions-Tracking (x, y aus MuJoCo)
        - Joint-Namen für Bewegungssteuerung
        - Reward-Tracking für Reinforcement Learning
        - Lokale Collision-Maps
        - Pfad-Speicher für A*
        
        Args:
            env: Referenz zum Environment
            mu_index: Body-ID in MuJoCo (91, 99, 105, 110, 115)
            mu_start: Startposition [x, y] in Metern
            mu_joint: Joint-Name Prefix (z.B. "slide_joint1")
            mu_start_move: Initiale Bewegungsrichtung [x, y]
            follow: True wenn dieser Mover Mover 0 folgen soll
            max_dist: Maximaler erlaubter Abstand zu anderen
            vel: Geschwindigkeitsfaktor
            cable_connect: Liste der verbundenen Kabel-IDs
            cable_start_mu: Body-IDs der Kabel-Startpunkte
        """
        # ========== REWARD TRACKING ==========
        self.reward_total = 0
        self.mean_reward = 0
        self.reward_sum = 0
        self.reward = 0
        self.done = False
        self.reward_list = []
        
        # ========== KOORDINATEN TRACKING ==========
        self.coord_list = []
        self.coords = []
        self.coords_x = []
        self.coords_y = []
        self.best_r = []
        
        # ========== ENVIRONMENT REFERENZ ==========
        self.env = env
        
        # ========== MOVER EIGENSCHAFTEN ==========
        self.mu_index = mu_index
        self.mu_start = mu_start
        self.x = mu_start[0]
        self.y = mu_start[1]
        
        # ========== JOINT KONTROLLE ==========
        mu_joint_x = mu_joint + "x"
        self.joint_x = mu_joint_x
        mu_joint_y = mu_joint + "y"
        self.joint_y = mu_joint_y
        
        # ========== BEWEGUNGSPARAMETER ==========
        self.vel = vel
        self.start_move = mu_start_move
        self.follow = follow
        self.max_dist = max_dist
        
        # ========== KABEL VERBINDUNGEN ==========
        self.cable_connect = cable_connect
        self.cable_start_mu = cable_start_mu
        
        # ========== WINKEL-KOORDINATION ==========
        self.action_phi1 = [0, 0]
        self.action_phi2 = [0, 0]
        self.phi1 = 0
        self.phi2 = 0
        
        # ========== LOKALE COLLISION MAPS ==========
        self.mu_collision_map = np.zeros((5, 5))
        self.mu_cable_collision_map = np.zeros((7, 7))
        
        # ========== ZIEL KOORDINATEN ==========
        self.x_t = 0
        self.y_t = 0

        # ========== PFAD SPEICHER ==========
        self.path = [] 
        self.path_original = []

        # ========== WP-Control ==========
        self.wp_reached = False

    def update_pos(self):
        """
        Aktualisiert Position aus MuJoCo-Daten.
        Wird jeden Simulationsschritt aufgerufen.
        """
        self.x = self.env.data.xpos[self.mu_index][0]
        self.y = self.env.data.xpos[self.mu_index][1]

    def get_distance(self, x, y, dist_norm=0):
        """
        Berechnet Distanz zu einem Punkt.
        
        Was hier berechnet wird:
        - Euklidische Distanz mit Pythagoras
        - Optional: Normalisierung auf [-1, 1]
        
        Args:
            x, y: Zielpunkt
            dist_norm: Normalisierungsfaktor (0 = keine Normalisierung)
        
        Returns:
            Distanz oder normalisierte Distanz
        """
        dist = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        
        if dist_norm > 0:
            return (dist - dist_norm/2) / (dist_norm/2)
        else:
            return dist
    
    def get_distance_x(self, x):
        """X-Distanz (mit Vorzeichen)"""
        return self.x - x
    
    def get_distance_y(self, y):
        """Y-Distanz (mit Vorzeichen)"""
        return self.y - y
    
    def get_distance_target(self, norm=True):
        """
        Distanz zum Ziel.
        
        Args:
            norm: True für Normalisierung auf [-1, 1]
        """
        dist = math.sqrt((self.x - self.x_t)**2 + (self.y - self.y_t)**2)
        
        if norm:
            return (dist - 5/2) / (5/2)
        else:
            return dist

    def get_angle(self, x, y, norm=True):
        """
        Winkel zu einem Punkt.
        
        Verwendet atan2 für vollständigen Winkelbereich [-π, π].
        
        Args:
            x, y: Zielpunkt
            norm: True für Normalisierung auf [-1, 1]
        """
        angle = math.atan2((self.y - y), (self.x - x))
        
        if norm:
            return angle / 3.142
        else:
            return angle
    
    def get_angle_target(self, norm=True):
        """Winkel zum Ziel"""
        angle = math.atan2((self.y - self.y_t), (self.x - self.x_t))
        
        if norm:
            return angle / 3.142
        else:
            return angle
    
    def make_move(self, action):
        """
        Setzt die Geschwindigkeit der Joints.
        
        Was hier passiert:
        - Multipliziert Action mit Geschwindigkeitsfaktor
        - Setzt Joint-Velocities in MuJoCo
        
        Args:
            action: [x, y] Bewegungsrichtung (normalisiert)
        """
        self.env.data.joint(self.joint_x).qvel[0] = self.vel * action[0]
        self.env.data.joint(self.joint_y).qvel[0] = self.vel * action[1]

    def set_target(self, x_t, y_t):
        """Setzt neue Zielkoordinaten"""
        self.x_t = x_t
        self.y_t = y_t

    def choose_constraint_action(self, step, dist):
        """
        Wählt Aktion basierend auf Constraints.
        
        Prioritäten-Reihenfolge:
        1. Follow-Constraint: Abstand zu Mover 0 einhalten
        2. Abstand-Constraint: Max-Abstand zu anderen
        3. Kollisions-Vermeidung: Lokale Hindernisse
        4. Winkel-Koordination: Bei ähnlichen Winkeln
        
        Returns:
            [x, y] Action oder [0, 0] wenn keine Constraint-Action
        """
        # ========== CONSTRAINT 1: FOLLOW ==========
        if self.follow and dist > self.max_dist:
            x_dist = self.get_distance_x(self.env.movers[0].x)
            y_dist = self.get_distance_y(self.env.movers[0].y)
            action = [-x_dist/dist, -y_dist/dist]
            return action
        
        # ========== CONSTRAINT 2: ABSTAND ==========
        if not self.follow:
            for i in range(self.env.num_agents - 1):
                dist1 = self.get_distance(self.env.movers[i + 1].x, self.env.movers[i + 1].y)
                if dist1 > self.env.movers[i + 1].max_dist:
                    x_dist = self.get_distance_x(self.env.movers[i + 1].x)
                    y_dist = self.get_distance_y(self.env.movers[i + 1].y)
                    action = [-x_dist/dist1, -y_dist/dist1]
                    return action
        
        # ========== CONSTRAINT 3: KOLLISION ==========
        if np.sum(self.mu_collision_map) > 1:
            action = self.collision_avoidance()
            return action
        
        # # ========== CONSTRAINT 4: WINKEL-KOORDINATION ==========
        # if not np.array_equal(self.action_phi2, [0, 0]):
        #     return self.action_phi2
        
        # if not np.array_equal(self.action_phi1, [0, 0]):
        #     return self.action_phi1
        
        # Keine Constraint-Action
        return [0, 0]

    def lokal_collision_map(self):
        """
        Erstellt lokale Collision-Maps um den Mover.
        
        Was hier passiert:
        1. 5×5 Map für direkte Kollisionen
        2. 7×7 Map für Kabel-Kollisionen
        3. Filtert eigene Kabel und eigenen Body raus
        4. Markiert Wände am Rand
        """
        self.update_pos()
        
        x_idx = int(round(self.x, 1) * 10)
        y_idx = int(round(self.y, 1) * 10)

        # ========== 5×5 MAP ==========
        self.mu_collision_map = np.zeros((5, 5))

        for i in range(5):
            for j in range(5):
                global_y = y_idx - 2 + i
                global_x = x_idx - 2 + j
                
                if 0 < global_y < 38 and 0 < global_x < 68:
                    entry = self.env.collision_map[global_y, global_x]
                    
                    # if entry in self.cable_connect or entry == self.mu_index:
                    if entry == self.mu_index:
                        self.mu_collision_map[i, j] = 0
                    elif entry > 0:
                        self.mu_collision_map[i, j] = 1
                else:
                    self.mu_collision_map[i, j] = 1

        # ========== 7×7 MAP ==========
        self.mu_cable_collision_map = np.zeros((7, 7))

        for i in range(7):
            for j in range(7):
                global_y = y_idx - 3 + i
                global_x = x_idx - 3 + j
                
                if 0 < global_y < 48 and 0 < global_x < 68:
                    entry = self.env.collision_map[global_y, global_x]
                    
                    if entry == self.mu_index or entry == 0:
                        self.mu_cable_collision_map[i, j] = 0
                    else:
                        self.mu_cable_collision_map[i, j] = 1
                else:
                    self.mu_cable_collision_map[i, j] = 1
    
    def collision_avoidance(self):
        """
        Einfache Kollisionsvermeidung.
        
        Strategie:
        - Bewege dich weg von der Seite mit mehr Hindernissen
        - Links vs Rechts und Oben vs Unten
        """
        x_dir = 0.5 if np.sum(self.mu_collision_map[:, :1]) > np.sum(self.mu_collision_map[:, 2:]) else -0.5
        y_dir = 0.5 if np.sum(self.mu_collision_map[:1, :]) > np.sum(self.mu_collision_map[2:, :]) else -0.5
        
        return [x_dir, y_dir]
    
    def deterministic_move_t(self):
        """
        Direkte Bewegung zum Ziel.
        
        Was hier passiert:
        1. Berechnet Richtungsvektor zum Ziel
        2. Normalisiert auf Manhattan-Distanz 0.5
        """
        x_dist = self.get_distance_x(self.x_t)
        y_dist = self.get_distance_y(self.y_t)
        
        norm = math.sqrt(x_dist ** 2 + y_dist ** 2)
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        scaling = 0.5 / (abs(x_dir) + abs(y_dir))
        x_scaled = x_dir * scaling
        y_scaled = y_dir * scaling

        return [x_scaled, y_scaled]
    
    def deterministic_move_w(self, w_x, w_y):
        """
        Direkte Bewegung zum Ziel.
        
        Was hier passiert:
        1. Berechnet Richtungsvektor zum Ziel
        2. Normalisiert auf Manhattan-Distanz 0.5
        """
        x_dist = self.get_distance_x(w_x)
        y_dist = self.get_distance_y(w_y)
        
        norm = math.sqrt(x_dist ** 2 + y_dist ** 2)
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        scaling = 0.5 / (abs(x_dir) + abs(y_dir))
        x_scaled = x_dir * scaling
        y_scaled = y_dir * scaling

        return [x_scaled, y_scaled]
    
    def diverted_deterministic_move_t(self, div):
        """
        Direkte Bewegung zum Ziel mit einer Ablenkung durch 'div'.
        div = 1.0 -> 45° Rechtskurve
        div = -1.0 -> 45° Linkskurve
        """
        # 1. Distanzen berechnen
        x_dist = self.get_distance_x(self.x_t)
        y_dist = self.get_distance_y(self.y_t)
        
        # 2. Richtungsvektor normalisieren
        norm = math.sqrt(x_dist ** 2 + y_dist ** 2)
        if norm == 0: return [0, 0] # Ziel erreicht
        
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        # 3. Rotation berechnen
        # Winkel in Radiant: 45 Grad sind pi/4
        # Da y in vielen Koordinatensystemen nach unten zunimmt, 
        # muss man bei der Drehrichtung ggf. das Vorzeichen prüfen.
        angle = div * (math.pi / 4)
        
        # Rotationsmatrix anwenden:
        # x' = x * cos(a) - y * sin(a)
        # y' = x * sin(a) + y * cos(a)
        x_rotated = x_dir * math.cos(angle) - y_dir * math.sin(angle)
        y_rotated = x_dir * math.sin(angle) + y_dir * math.cos(angle)

        # 4. Skalierung auf Manhattan-Distanz 0.5
        scaling = 0.5 / (abs(x_rotated) + abs(y_rotated))
        x_scaled = x_rotated * scaling
        y_scaled = y_rotated * scaling

        return [x_scaled, y_scaled]
    
    def damped_diverted_deterministic_move_t(self, div):
        """
        Direkte Bewegung zum Ziel mit kontinuierlicher Dämpfung.
        Die Ablenkung ist bei großer Distanz maximal und nähert sich 
        bei Annäherung an das Ziel asymptotisch der Null an.
        """
        # 1. Distanzen berechnen
        x_dist = self.get_distance_x(self.x_t)
        y_dist = self.get_distance_y(self.y_t)
        
        # 2. Distanz (Norm) berechnen
        norm = math.sqrt(x_dist ** 2 + y_dist ** 2)
        
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        # 3. Kontinuierliche Dämpfung
        # Wir nutzen eine Sättigungsfunktion. 
        # Je größer 'norm', desto näher ist der Faktor an 1.0.
        # 'k' steuert, wie schnell die Ablenkung bei Annäherung verschwindet.
        k = 0.1  # Ein Dämpfungskoeffizient (experimentell anpassbar)
        damping_factor = 1 - math.exp(-k * norm)
        
        # Alternativ sehr simpel: damping_factor = norm / (norm + 5)
        
        angle = div * (math.pi / 4) * damping_factor
        
        # 4. Rotationsmatrix anwenden
        x_rotated = x_dir * math.cos(angle) - y_dir * math.sin(angle)
        y_rotated = x_dir * math.sin(angle) + y_dir * math.cos(angle)

        # 5. Skalierung auf Manhattan-Distanz 0.5
        scaling = 0.5 / (abs(x_rotated) + abs(y_rotated))
        x_scaled = x_rotated * scaling
        y_scaled = y_rotated * scaling

        return [x_scaled, y_scaled]
    
    def direct_move(self):
        """
        Prüft ob direkter Weg zum Ziel frei ist.
        
        Strategie:
        - Prüft Punkte entlang der Linie zum Ziel
        - Mehr Punkte bei größerer Distanz
        - Stoppt bei erstem Hindernis
        
        Returns:
            True wenn Weg frei, sonst False
        """
        self.update_pos()
        
        dist_to_target = self.get_distance_target(False)
        steps = max(10, int(dist_to_target * 15))
        
        for i in range(1, steps):
            t = i / float(steps)
            check_x = self.x + t * (self.x_t - self.x)
            check_y = self.y + t * (self.y_t - self.y)
            
            x_idx = int(round(check_x, 1) * 10)
            y_idx = int(round(check_y, 1) * 10)
            
            if x_idx < 0 or x_idx >= 72 or y_idx < 0 or y_idx >= 50:
                return False
            
            if self.env.collision_map[y_idx, x_idx] > 0:
                if (self.env.collision_map[y_idx, x_idx] != self.mu_index and 
                    self.env.collision_map[y_idx, x_idx] not in self.cable_connect):
                    return False
        
        return True

    def apf_action(self):
        """
        Artificial Potential Fields (APF) für direkte Bewegung.
        Kombiniert:
        - Anziehung zum Ziel (attractive force)
        - Abstoßung von Hindernissen (repulsive force)
        
        Returns:
            action: [dx, dy] normalisierter Bewegungsvektor
        """
        # ========== ANZIEHUNGSKRAFT ZUM ZIEL ==========
        goal_vector = np.array([self.x_t - self.x, self.y_t - self.y])
        goal_distance = np.linalg.norm(goal_vector)
        
        if goal_distance > 0:
            attractive_force = (goal_vector / goal_distance) * min(1.0, goal_distance)
        else:
            attractive_force = np.array([0.0, 0.0])
        
        # ========== ABSTOSSUNGSKRAFT VON HINDERNISSEN ==========
        repulsive_force = np.array([0.0, 0.0])
        
        influence_radius = 0.5
        repulsion_weight = 0.8
        
        x_idx = int(round(self.x, 1) * 10)
        y_idx = int(round(self.y, 1) * 10)
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                map_y = dy + 2
                map_x = dx + 2
                
                if 0 <= map_y < 5 and 0 <= map_x < 5:
                    obstacle_id = self.mu_collision_map[map_y, map_x]
                    
                    if obstacle_id > 0:
                        obstacle_x = (x_idx + dx) / 10.0
                        obstacle_y = (y_idx + dy) / 10.0
                        
                        repulsion_vector = np.array([self.x - obstacle_x, self.y - obstacle_y])
                        distance = np.linalg.norm(repulsion_vector)
                        
                        if distance > 0.01 and distance < influence_radius:
                            repulsion_strength = repulsion_weight * (influence_radius - distance) / (distance ** 2)
                            repulsion_strength = min(repulsion_strength, 3.0)
                            repulsive_force += (repulsion_vector / distance) * repulsion_strength
        
        # ========== KOMBINIERE KRÄFTE ==========
        total_force = attractive_force + repulsive_force
        
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > 0:
            action = total_force / force_magnitude
            
            if goal_distance < 0.3:
                speed_factor = max(0.3, goal_distance / 0.3)
                action = action * speed_factor
        else:
            action = np.array([0.0, 0.0])
        
        return action
    
    def astar_move_to_target(self):
        """Eine Action aus A* (auf env.collision_map) Richtung (self.x_t, self.y_t)."""
        self.update_pos()
        return self._astar_move(self.x_t, self.y_t)

    def astar_move_to_waypoint(self, w_x, w_y):
        """Eine Action aus A* (auf env.collision_map) Richtung Wegpunkt (w_x, w_y)."""
        self.update_pos()
        return self._astar_move(w_x, w_y)

    # -------------------------
    # interne Mini-Helfer
    # -------------------------
    def _astar_move(self, goal_x, goal_y):
        # world -> grid
        sx = int(round(self.x, 1) * 10)
        sy = int(round(self.y, 1) * 10)
        gx = int(round(goal_x, 1) * 10)
        gy = int(round(goal_y, 1) * 10)

        # A*: nächste Zelle bestimmen
        nxt = self._astar_next_cell((sx, sy), (gx, gy))
        if nxt is None:
            # fallback: wie deterministic_move (direkt)
            return self.deterministic_move_w(goal_x, goal_y)

        # grid -> world (nächster Schritt)
        nx, ny = nxt
        wx, wy = nx / 10.0, ny / 10.0

        # Action wie deterministic_move: auf Manhattan 0.5 skalieren
        x_dist = self.get_distance_x(wx)
        y_dist = self.get_distance_y(wy)
        norm = math.sqrt(x_dist * x_dist + y_dist * y_dist)
        if norm < 1e-9:
            return [0, 0]

        x_dir = -x_dist / norm
        y_dir = -y_dist / norm
        scaling = 0.5 / (abs(x_dir) + abs(y_dir) + 1e-9)
        return [x_dir * scaling, y_dir * scaling]

    def _free_cell(self, x, y):
        # bounds
        if x < 0 or x >= 72 or y < 0 or y >= 50:
            return False

        v = self.env.collision_map[y, x]
        # frei: 0 oder eigener mover oder eigene kabel
        if v == 0 or v == self.mu_index or (v in self.cable_connect):
            return True
        return False

    def _astar_next_cell(self, start, goal):
        """Sehr simples A* (4er Nachbarschaft). Gibt die nächste Zelle (x,y) oder None."""
        if start == goal:
            return None

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_pq = [(h(start, goal), 0, start)]
        came = {start: None}
        gscore = {start: 0}

        while open_pq:
            _, g, cur = heapq.heappop(open_pq)

            if cur == goal:
                # Pfad rekonstruieren -> erster Schritt
                path = []
                n = cur
                while n is not None:
                    path.append(n)
                    n = came[n]
                path.reverse()
                return path[1] if len(path) > 1 else None

            x, y = cur
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                if not self._free_cell(nx, ny):
                    continue
                ng = g + 1
                if (nx, ny) not in gscore or ng < gscore[(nx, ny)]:
                    gscore[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    heapq.heappush(open_pq, (ng + h((nx, ny), goal), ng, (nx, ny)))

        return None

    def squeeze_action_r(self, dist, range):
        x_dist = self.get_distance_x(self.env.movers[0].x)
        y_dist = self.get_distance_y(self.env.movers[0].y)

        if dist / self.max_dist > range:
            action = [-x_dist/dist, -y_dist/dist]
            return action
        else:
            action = [x_dist/dist, y_dist/dist]
            return action
        
    def squeeze_action(self, dist, sq):
        x_dist = self.get_distance_x(self.env.movers[0].x)
        y_dist = self.get_distance_y(self.env.movers[0].y)

        if sq:
            action = [-x_dist/dist, -y_dist/dist]
            return action
        else:
            action = [x_dist/dist, y_dist/dist]
            return action
        
    def orbit_mover_0(self, radius, clockwise, speed=1.0):
        """
        Berechnet eine Action, um in einem festen Radius um Mover 0 zu rotieren.
        
        Args:
            radius: Der gewünschte Abstand zu Mover 0.
            speed: Geschwindigkeit der Rotation.
            clockwise: Drehrichtung.
            
        Returns:
            [x_dir, y_dir]: Bewegungsvektor für make_move().
        """
        # 1. Aktuellen Vektor von Mover 0 zu diesem Mover bestimmen
        dx = self.x - self.env.movers[0].x
        dy = self.y - self.env.movers[0].y
        current_dist = math.sqrt(dx**2 + dy**2)

        # 2. Normalisierter Richtungsvektor (radial)
        nx = dx / current_dist
        ny = dy / current_dist

        # 3. Tangential-Vektor berechnen (Die Richtung "entlang" des Kreises)
        # Ein Vektor (x, y) wird durch (-y, x) um 90° gedreht
        if clockwise:
            tx, ty = ny, -nx
        else:
            tx, ty = -ny, nx

        # 4. Korrektur-Vektor (Zentripetalkraft)
        # Wenn der Mover zu weit weg oder zu nah dran ist, lenken wir ihn zum Radius zurück
        dist_error = radius - current_dist
        # Wir kombinieren Tangente (Rotation) und Normale (Radius-Korrektur)
        
        x_dir = tx * speed + nx * dist_error
        y_dir = ty * speed + ny * dist_error

        # 5. Normalisieren für gleichmäßige Geschwindigkeit
        norm = math.sqrt(x_dir**2 + y_dir**2)
        return [x_dir / norm, y_dir / norm]