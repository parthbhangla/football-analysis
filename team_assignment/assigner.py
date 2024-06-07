# Import Libraries
from sklearn.cluster import KMeans

class TeamAssigner:

    def __init__(self):

        self.team_colors = {}
        self.player_team_dict = {} # player_id: team_id

    # Clustering Model
    def get_clustering_model(self, image):

        # Reshaping the image into a 2D array of pixels
        image_2d = image.reshape(-1, 3)

        # Perfomring KMeans Clustering for 2 clusters
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)

        return kmeans
    
    # Getting the player color
    def get_player_color(self, frame, bbox):

        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Taking top-half of image (see-visualization)
        top_half_image = image[0:int(image.shape[0]/2),:]
        kmeans = self.get_clustering_model(top_half_image)

        # Labels for each pixel
        labels = kmeans.labels_

        # Reshaping the clustered image back to original dimensions
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Getting the player cluster (see-visualization)
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color
    
    # Assigning a team color
    def assign_team_color(self, frame, player_detections):
        
        player_colors = []
        for _ , player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Dividing the colors into 2 team
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)
        
        # Saving this for later
        self.kmeans = kmeans

        # Assigning team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    # Getting a player's team
    def get_player_team(self, frame, player_bbox, player_id):

        # Using the stored data in dictionary if available
        if player_id  in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
         
        # Predicting team id - returns 0 or 1
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] 

        # To Keep the team id between 1 and 2
        team_id = team_id + 1

        # Storing this for later
        self.player_team_dict[player_id] = team_id

        return team_id   
