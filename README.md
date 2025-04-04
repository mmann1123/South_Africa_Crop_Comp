# South_Africa_Crop_Comp
https://platform.ai4eo.eu/ai4food-security-south-africa

https://mlhub.earth/

https://mlhub.earth/data/ref_fusion_competition_south_africa


## 🧪 Submitting Your Prediction

To participate, submit a `prediction.csv` file in the following format:

```
crop_name
Lucerne/Medics
Small grain grazing
Barley
Lucerne/Medics
Small grain grazing
Canola
Wheat
```

 
### 📋 Submission Requirements

- File **must be named** `prediction.csv`
- Include **only one column** with header `crop_name`
- Predictions must be **ordered by field ID** (same order as the provided test data) - or the order of the test data in the `test` folder of the geojson file geometries
- Save your file in:

```
submissions/prediction.csv
```

### ⚙️ What Happens Next

Once your pull request is submitted:
- A GitHub Action will automatically run
- It will compare your predictions to a hidden ground truth
- You’ll receive a **Cohen’s Kappa**, **F1 score**, and **Cross Entropy with binary outcome for each crop** as a comment on your PR
Good luck! 🌾📈

Where Cross Entropy with binary outcome for each crop is calculated as:

$J = -\frac{1}{N} \left( \sum_{i=1}^{9} y_{j,i} \ln(p_{j,i}) \right)$


## 🌾 Original Harvest Challenge Leaderboard (2023)

| 🏅 Position | 👥 Team         | 🎯 Cross Entropy Score |
|-------------|----------------|------------------------|
| 1           | AI4EO          | 3.588                  |
| 2           | EagleEyes      | 3.601                  |
| 3           | MEOTEQ         | 3.714                  |
| 4           | Panopterra     | 3.848                  |
| 5           | Adrián Cal     | 3.955                  |
| 6           | Harvest UMD    | 3.981                  |
| 7           | N & S          | 3.981                  |

_Source: [TCSA-AI Leaderboard](https://platform.ai4eo.eu/ai4food-security-south-africa/leaderboard)_



More on [entropy scoring here](https://github.com/radiantearth/spot-the-crop-challenge)