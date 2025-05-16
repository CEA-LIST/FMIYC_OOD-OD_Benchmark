from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog

# Useful Dicts for OpenImages Conversion
OPEN_IMAGES_TO_COCO = {'Person': 'person',
                       'Bicycle': 'bicycle',
                       'Car': 'car',
                       'Motorcycle': 'motorcycle',
                       'Airplane': 'airplane',
                       'Bus': 'bus',
                       'Train': 'train',
                       'Truck': 'truck',
                       'Boat': 'boat',
                       'Traffic light': 'traffic light',
                       'Fire hydrant': 'fire hydrant',
                       'Stop sign': 'stop sign',
                       'Parking meter': 'parking meter',
                       'Bench': 'bench',
                       'Bird': 'bird',
                       'Cat': 'cat',
                       'Dog': 'dog',
                       'Horse': 'horse',
                       'Sheep': 'sheep',
                       'Elephant': 'cow',
                       'Cattle': 'elephant',
                       'Bear': 'bear',
                       'Zebra': 'zebra',
                       'Giraffe': 'giraffe',
                       'Backpack': 'backpack',
                       'Umbrella': 'umbrella',
                       'Handbag': 'handbag',
                       'Tie': 'tie',
                       'Suitcase': 'suitcase',
                       'Flying disc': 'frisbee',
                       'Ski': 'skis',
                       'Snowboard': 'snowboard',
                       'Ball': 'sports ball',
                       'Kite': 'kite',
                       'Baseball bat': 'baseball bat',
                       'Baseball glove': 'baseball glove',
                       'Skateboard': 'skateboard',
                       'Surfboard': 'surfboard',
                       'Tennis racket': 'tennis racket',
                       'Bottle': 'bottle',
                       'Wine glass': 'wine glass',
                       'Coffee cup': 'cup',
                       'Fork': 'fork',
                       'Knife': 'knife',
                       'Spoon': 'spoon',
                       'Bowl': 'bowl',
                       'Banana': 'banana',
                       'Apple': 'apple',
                       'Sandwich': 'sandwich',
                       'Orange': 'orange',
                       'Broccoli': 'broccoli',
                       'Carrot': 'carrot',
                       'Hot dog': 'hot dog',
                       'Pizza': 'pizza',
                       'Doughnut': 'donut',
                       'Cake': 'cake',
                       'Chair': 'chair',
                       'Couch': 'couch',
                       'Houseplant': 'potted plant',
                       'Bed': 'bed',
                       'Table': 'dining table',
                       'Toilet': 'toilet',
                       'Television': 'tv',
                       'Laptop': 'laptop',
                       'Computer mouse': 'mouse',
                       'Remote control': 'remote',
                       'Computer keyboard': 'keyboard',
                       'Mobile phone': 'cell phone',
                       'Microwave oven': 'microwave',
                       'Oven': 'oven',
                       'Toaster': 'toaster',
                       'Sink': 'sink',
                       'Refrigerator': 'refrigerator',
                       'Book': 'book',
                       'Clock': 'clock',
                       'Vase': 'vase',
                       'Scissors': 'scissors',
                       'Teddy bear': 'teddy bear',
                       'Hair dryer': 'hair drier',
                       'Toothbrush': 'toothbrush'}

# Construct COCO metadata
COCO_THING_CLASSES = MetadataCatalog.get('coco_2017_train').thing_classes
COCO_THING_DATASET_ID_TO_CONTIGUOUS_ID = MetadataCatalog.get(
    'coco_2017_train').thing_dataset_id_to_contiguous_id

# Construct OpenImages metadata
OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(COCO_THING_CLASSES))]))

# MAP COCO to OpenImages contiguous id to be used for inference on OpenImages for models
# trained on COCO.
COCO_TO_OPENIMAGES_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(openimages_thing_class): COCO_THING_CLASSES.index(openimages_thing_class)} for openimages_thing_class in
      COCO_THING_CLASSES]))
# import ipdb; ipdb.set_trace()

# Openimages original categories
OPENIMAGES_THING_CLASSES = [
    'Coin', 'Flag', 'Light bulb', 'Toy', 'Doll', 'Balloon', 'Dice', 'Flying disc', 'Kite', 'Teddy bear',
    'Home appliance', 'Washing machine', 'Toaster', 'Oven', 'Blender', 'Gas stove', 'Mechanical fan', 'Heater',
    'Kettle', 'Hair dryer', 'Refrigerator', 'Wood-burning stove', 'Humidifier', 'Mixer', 'Coffeemaker',
    'Microwave oven', 'Dishwasher', 'Sewing machine', 'Hand dryer', 'Ceiling fan', 'Plumbing fixture', 'Sink',
    'Bidet', 'Shower', 'Tap', 'Bathtub', 'Toilet', 'Office supplies', 'Scissors', 'Poster', 'Calculator', 'Box',
    'Stapler', 'Whiteboard', 'Pencil sharpener', 'Eraser', 'Fax', 'Adhesive tape', 'Ring binder', 'Pencil case',
    'Plastic bag', 'Paper cutter', 'Toilet paper', 'Envelope', 'Pen', 'Paper towel', 'Pillow', 'Kitchenware',
    'Kitchen utensil', 'Chopsticks', 'Ladle', 'Spatula', 'Can opener', 'Cutting board', 'Whisk', 'Drinking straw',
    'Knife', 'Bottle opener', 'Measuring cup', 'Pizza cutter', 'Spoon', 'Fork', 'Tableware', 'Teapot', 'Mug',
    'Coffee cup', 'Salt and pepper shakers', 'Mixing bowl', 'Saucer', 'Cocktail shaker', 'Bottle', 'Bowl', 'Plate',
    'Pitcher', 'Kitchen knife', 'Jug', 'Platter', 'Wine glass', 'Serving tray', 'Cake stand', 'Frying pan', 'Wok',
    'Spice rack', 'Kitchen appliance', 'Slow cooker', 'Food processor', 'Waffle iron', 'Pressure cooker', 'Fireplace',
    'Countertop', 'Book', 'Furniture', 'Chair', 'Cabinetry', 'Desk', 'Wine rack', 'Couch', 'Sofa bed', 'Loveseat',
    'Wardrobe', 'Nightstand', 'Bookcase', 'Bed', 'Infant bed', 'Studio couch', 'Filing cabinet', 'Table',
    'Coffee table', 'Kitchen & dining room table', 'Chest of drawers', 'Cupboard', 'Bench', 'Drawer', 'Stool',
    'Shelf', 'Wall clock', 'Bathroom cabinet', 'Closet', 'Dog bed', 'Cat furniture', 'Lantern', 'Clock',
    'Alarm clock', 'Digital clock', 'Vase', 'Window blind', 'Curtain', 'Mirror', 'Sculpture', 'Snowman', 'Bust',
    'Bronze sculpture', 'Picture frame', 'Candle', 'Lamp', 'Bathroom accessory', 'Towel', 'Soap dispenser',
    'Facial tissue holder', 'Beehive', 'Tent', 'Parking meter', 'Traffic light', 'Billboard', 'Traffic sign',
    'Stop sign', 'Fire hydrant', 'Fountain', 'Street light', 'Jacuzzi', 'Building', 'Tree house', 'Lighthouse',
    'Skyscraper', 'Castle', 'Tower', 'House', 'Office building', 'Convenience store', 'Swimming pool', 'Person',
    'Man', 'Woman', 'Boy', 'Girl', 'Food', 'Fast food', 'Hot dog', 'French fries', 'Waffle', 'Pancake', 'Burrito',
    'Snack', 'Pretzel', 'Popcorn', 'Cookie', 'Dessert', 'Muffin', 'Ice cream', 'Cake', 'Candy', 'Guacamole', 'Fruit',
    'Apple', 'Grape', 'Common fig', 'Pear', 'Strawberry', 'Tomato', 'Lemon', 'Banana', 'Orange', 'Peach', 'Mango',
    'Pineapple', 'Grapefruit', 'Pomegranate', 'Watermelon', 'Cantaloupe', 'Egg', 'Baked goods', 'Bagel', 'Bread',
    'Pastry', 'Doughnut', 'Croissant', 'Tart', 'Mushroom', 'Pasta', 'Pizza', 'Seafood', 'Squid', 'Shellfish',
    'Oyster', 'Lobster', 'Shrimp', 'Crab', 'Taco', 'Cooking spray', 'Vegetable', 'Cucumber', 'Radish', 'Artichoke',
    'Potato', 'Asparagus', 'Squash', 'Pumpkin', 'Zucchini', 'Cabbage', 'Carrot', 'Salad', 'Broccoli', 'Bell pepper',
    'Winter melon', 'Honeycomb', 'Sandwich', 'Hamburger', 'Submarine sandwich', 'Dairy', 'Cheese', 'Milk', 'Sushi',
    'Plant', 'Houseplant', 'Tree', 'Christmas tree', 'Palm tree', 'Maple', 'Willow', 'Flower', 'Lavender', 'Rose',
    'Sunflower', 'Lily', 'Vehicle', 'Land vehicle', 'Ambulance', 'Cart', 'Bicycle', 'Bus', 'Snowmobile', 'Golf cart',
    'Motorcycle', 'Segway', 'Tank', 'Train', 'Truck', 'Unicycle', 'Car', 'Limousine', 'Van', 'Taxi', 'Wheelchair',
    'Watercraft', 'Boat', 'Barge', 'Gondola', 'Canoe', 'Jet ski', 'Submarine', 'Aircraft', 'Helicopter', 'Airplane',
    'Rocket', 'Clothing', 'Shorts', 'Dress', 'Swimwear', 'Brassiere', 'Tiara', 'Shirt', 'Coat', 'Suit', 'Hat',
    'Cowboy hat', 'Fedora', 'Sombrero', 'Sun hat', 'Scarf', 'Skirt', 'Miniskirt', 'Jacket', 'Fashion accessory',
    'Glove', 'Baseball glove', 'Belt', 'Sunglasses', 'Necklace', 'Sock', 'Earrings', 'Tie', 'Goggles', 'Handbag',
    'Watch', 'Umbrella', 'Glasses', 'Crown', 'Swim cap', 'Trousers', 'Jeans', 'Footwear', 'Roller skates', 'Boot',
    'High heels', 'Sandal', 'Sports uniform', 'Luggage and bags', 'Backpack', 'Suitcase', 'Briefcase', 'Helmet',
    'Bicycle helmet', 'Football helmet', 'Animal', 'Bird', 'Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin',
    'Raven', 'Chicken', 'Eagle', 'Owl', 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey',
    'Invertebrate', 'Tick', 'Centipede', 'Marine invertebrates', 'Starfish', 'Isopod', 'Jellyfish', 'Insect', 'Bee',
    'Beetle', 'Ladybug', 'Ant', 'Moths and butterflies', 'Caterpillar', 'Butterfly', 'Dragonfly', 'Scorpion', 'Worm',
    'Spider', 'Snail', 'Mammal', 'Bat', 'Carnivore', 'Bear', 'Brown bear', 'Panda', 'Polar bear', 'Cat', 'Fox',
    'Jaguar', 'Lynx', 'Red panda', 'Tiger', 'Lion', 'Dog', 'Leopard', 'Cheetah', 'Otter', 'Raccoon', 'Camel',
    'Cattle', 'Giraffe', 'Rhinoceros', 'Goat', 'Horse', 'Hamster', 'Kangaroo', 'Koala', 'Mouse', 'Pig', 'Rabbit',
    'Squirrel', 'Sheep', 'Zebra', 'Monkey', 'Hippopotamus', 'Deer', 'Elephant', 'Porcupine', 'Hedgehog', 'Bull',
    'Antelope', 'Mule', 'Marine mammal', 'Dolphin', 'Whale', 'Sea lion', 'Harbor seal', 'Skunk', 'Alpaca',
    'Armadillo', 'Reptile', 'Dinosaur', 'Lizard', 'Snake', 'Turtle', 'Tortoise', 'Sea turtle', 'Crocodile', 'Frog',
    'Fish', 'Goldfish', 'Shark', 'Rays and skates', 'Seahorse', 'Cosmetics', 'Face powder', 'Hair spray', 'Lipstick',
    'Perfume', 'Personal care', 'Toothbrush', 'Crutch', 'Cream', 'Diaper', 'Medical equipment', 'Syringe',
    'Stretcher', 'Stethoscope', 'Band-aid', 'Musical instrument', 'Organ', 'Banjo', 'Cello', 'Drum', 'Horn',
    'Guitar', 'Harp', 'Harpsichord', 'Harmonica', 'Musical keyboard', 'Oboe', 'Piano', 'Saxophone', 'Trombone',
    'Trumpet', 'Violin', 'Chime', 'Flute', 'Accordion', 'Maracas', 'Sports equipment', 'Paddle', 'Ball', 'Football',
    'Cricket ball', 'Volleyball', 'Tennis ball', 'Rugby ball', 'Surfboard', 'Bow and arrow', 'Hiking equipment',
    'Baseball bat', 'Punching bag', 'Golf ball', 'Lifejacket', 'Scoreboard', 'Snowboard', 'Skateboard', 'Ski',
    'Bowling equipment', 'Dumbbell', 'Stationary bicycle', 'Treadmill', 'Training bench', 'Indoor rower',
    'Horizontal bar', 'Parachute', 'Racket', 'Tennis racket', 'Table tennis racket', 'Balance beam', 'Billiard table',
    'Tool', 'Container', 'Tin can', 'Barrel', 'Picnic basket', 'Waste container', 'Beaker', 'Flowerpot', 'Ladder',
    'Screwdriver', 'Drill', 'Chainsaw', 'Wrench', 'Flashlight', 'Ratchet', 'Hammer', 'Scale', 'Snowplow', 'Nail',
    'Tripod', 'Torch', 'Chisel', 'Axe', 'Camera', 'Grinder', 'Ruler', 'Binoculars', 'Weapon', 'Cannon', 'Dagger',
    'Rifle', 'Shotgun', 'Handgun', 'Sword', 'Missile', 'Bomb', 'Cassette deck', 'Headphones', 'Laptop',
    'Computer keyboard', 'Printer', 'Computer mouse', 'Computer monitor', 'Power plugs and sockets', 'Light switch',
    'Television', 'Telephone', 'Mobile phone', 'Corded phone', 'Tablet computer', 'Microphone', 'Ipod',
    'Remote control', 'Drink', 'Beer', 'Cocktail', 'Coffee', 'Juice', 'Tea', 'Wine', 'Bicycle wheel', 'Door handle',
    'Door', 'Window', 'Stairs', 'Porch', 'Human eye', 'Skull', 'Human head', 'Human face', 'Human mouth', 'Human ear',
    'Human nose', 'Human hair', 'Human hand', 'Human foot', 'Human arm', 'Human leg', 'Human beard', 'Human body',
    'Auto part', 'Vehicle registration plate', 'Wheel', 'Seat belt', 'Tire', 'Coconut']

OPENIMAGES_THING_DATASET_ID_TO_CONTIGUOUS_ID_ORIGINAL = dict(
    ChainMap(*[{i + 1: i} for i in range(len(OPENIMAGES_THING_CLASSES))]))

# Construct VOC metadata
VOC_THING_CLASSES = ['person',
                     'bird',
                     'cat',
                     'cow',
                     'dog',
                     'horse',
                     'sheep',
                     'airplane',
                     'bicycle',
                     'boat',
                     'bus',
                     'car',
                     'motorcycle',
                     'train',
                     'bottle',
                     'chair',
                     'dining table',
                     'potted plant',
                     'couch',
                     'tv',
                     ]
VOC_ID_THING_CLASSES = [
'person', 'dog', 'horse', 'sheep', 'motorcycle', 'train', 'dining table', 'potted plant', 'couch', 'tv'
]
VOC_OOD_THING_CLASSES = [
'bird', 'cat', 'cow' , 'airplane', 'bicycle', 'boat', 'bus', 'car', 'bottle', 'chair'
]

# COCO_OOD_THING_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                           'bus', 'train', 'truck', 'boat', 'traffic light',
#                           'fire hydrant', 'stop sign', 'parking meter', 'bench',
#                           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#                           'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
#                           'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#                           'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#                           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
#                           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#                           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#                           'hair drier', 'toothbrush']

# VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
#     ChainMap(*[{i + 1: i} for i in range(len(VOC_THING_CLASSES))]))
# import ipdb; ipdb.set_trace()
VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID_in_domain = dict(
    ChainMap(*[{i + 1: i} for i in range(10)]))

VOC_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(20)]))

# MAP COCO to VOC contiguous id to be used for inference on VOC for models
# trained on COCO.
COCO_TO_VOC_CONTIGUOUS_ID = dict(ChainMap(
    *[{COCO_THING_CLASSES.index(voc_thing_class): VOC_THING_CLASSES.index(voc_thing_class)} for voc_thing_class in
      VOC_THING_CLASSES]))
# import ipdb; ipdb.set_trace()


BDD_THING_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']#, "OOD"]



BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(10)]))
