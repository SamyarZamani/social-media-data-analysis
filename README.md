### Sentiment Classification from Persian Social Media Texts

در این پروژه تلاش شد احساسات پست‌های شبکه‌های اجتماعی شناسایی شود.

# **importing part**


```python
!pip install nltk --quiet
import os
import nltk
from google.colab import files
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer

# ساختار پوشه‌ها
os.makedirs("social-media-data-analysis/data", exist_ok=True)
os.makedirs("social-media-data-analysis/analysis", exist_ok=True)
os.makedirs("social-media-data-analysis/report", exist_ok=True)

# ساخت فایل‌های متنی خالی
open("social-media-data-analysis/README.md", "w").close()
open("social-media-data-analysis/requirements.txt", "w").close()

nltk.download('vader_lexicon')
```

    [nltk_data] Downloading package vader_lexicon to /root/nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!





    True




```python
uploaded = files.upload()  # فایل رو از سیستم انتخاب کن

# انتقال فایل به پوشه data
for filename in uploaded.keys():
    shutil.move(filename, f"social-media-data-analysis/data/{filename}")
```



     <input type="file" id="files-d8952211-2cf0-41b5-9f88-9861afb961d5" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-d8952211-2cf0-41b5-9f88-9861afb961d5">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving sentimentdataset.csv to sentimentdataset.csv


# **data info**


```python
# بارگذاری داده
df = pd.read_csv("social-media-data-analysis/data/sentimentdataset.csv")

# نمایش ۵ ردیف اول
print(df.head())
print('''
-----------------------------------------------------------------
''')
# گرفتن اطلاعات مثل جنس دیتا
print(df.info())
```

       Unnamed: 0.1  Unnamed: 0  \
    0             0           0   
    1             1           1   
    2             2           2   
    3             3           3   
    4             4           4   
    
                                                    Text    Sentiment  \
    0   Enjoying a beautiful day at the park!        ...   Positive     
    1   Traffic was terrible this morning.           ...   Negative     
    2   Just finished an amazing workout! 💪          ...   Positive     
    3   Excited about the upcoming weekend getaway!  ...   Positive     
    4   Trying out a new recipe for dinner tonight.  ...   Neutral      
    
                 Timestamp            User     Platform  \
    0  2023-01-15 12:30:00   User123          Twitter     
    1  2023-01-15 08:45:00   CommuterX        Twitter     
    2  2023-01-15 15:45:00   FitnessFan      Instagram    
    3  2023-01-15 18:20:00   AdventureX       Facebook    
    4  2023-01-15 19:55:00   ChefCook        Instagram    
    
                                         Hashtags  Retweets  Likes       Country  \
    0   #Nature #Park                                  15.0   30.0     USA         
    1   #Traffic #Morning                               5.0   10.0     Canada      
    2   #Fitness #Workout                              20.0   40.0   USA           
    3   #Travel #Adventure                              8.0   15.0     UK          
    4   #Cooking #Food                                 12.0   25.0    Australia    
    
       Year  Month  Day  Hour  
    0  2023      1   15    12  
    1  2023      1   15     8  
    2  2023      1   15    15  
    3  2023      1   15    18  
    4  2023      1   15    19  
    
    -----------------------------------------------------------------
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 732 entries, 0 to 731
    Data columns (total 15 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   Unnamed: 0.1  732 non-null    int64  
     1   Unnamed: 0    732 non-null    int64  
     2   Text          732 non-null    object 
     3   Sentiment     732 non-null    object 
     4   Timestamp     732 non-null    object 
     5   User          732 non-null    object 
     6   Platform      732 non-null    object 
     7   Hashtags      732 non-null    object 
     8   Retweets      732 non-null    float64
     9   Likes         732 non-null    float64
     10  Country       732 non-null    object 
     11  Year          732 non-null    int64  
     12  Month         732 non-null    int64  
     13  Day           732 non-null    int64  
     14  Hour          732 non-null    int64  
    dtypes: float64(2), int64(6), object(7)
    memory usage: 85.9+ KB
    None


#**Social Engagement**

میزان مشارکت کاربران را از طریق ویژگی‌های Retweets و Likes
و شناسایی محتوای محبوب و ترجیحات کاربران


```python
# تبدیل Likes و Retweets به عدد صحیح (برای مرتب‌سازی راحت‌تر)
df["Likes"] = df["Likes"].astype(int)
df["Retweets"] = df["Retweets"].astype(int)

# نمایش ۱۰ پست برتر از نظر لایک
top_liked = df.sort_values(by="Likes", ascending=False).head(10)
print("Top 10 Posts by Likes:")
display(top_liked[["Text", "Likes", "Retweets", "Hashtags", "Platform"]])

# نمایش ۱۰ پست برتر از نظر ریتوییت
top_retweeted = df.sort_values(by="Retweets", ascending=False).head(10)
print("Top 10 Posts by Retweets:")
display(top_retweeted[["Text", "Likes", "Retweets", "Hashtags", "Platform"]])
```

    Top 10 Posts by Likes:




  <div id="df-9124ccfa-0fa0-459d-a6c2-d850fba26429" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Likes</th>
      <th>Retweets</th>
      <th>Hashtags</th>
      <th>Platform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>570</th>
      <td>Underneath the city lights, the dancer express...</td>
      <td>80</td>
      <td>40</td>
      <td>#Mesmerizing #NightDancePerformance</td>
      <td>Twitter</td>
    </tr>
    <tr>
      <th>345</th>
      <td>Motivated to achieve fitness goals after an in...</td>
      <td>80</td>
      <td>40</td>
      <td>#Motivation #FitnessGoals</td>
      <td>Facebook</td>
    </tr>
    <tr>
      <th>368</th>
      <td>Elation over discovering a rare book in a quai...</td>
      <td>80</td>
      <td>40</td>
      <td>#Elation #RareBookDiscovery</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>560</th>
      <td>In the serene beauty of a sunset, nature unfol...</td>
      <td>80</td>
      <td>40</td>
      <td>#Tranquility #SunsetBeauty</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>382</th>
      <td>A sense of wonder at the vastness of the cosmo...</td>
      <td>80</td>
      <td>40</td>
      <td>#Wonder #StargazingAdventure</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>355</th>
      <td>Anticipation for an upcoming adventure in an e...</td>
      <td>80</td>
      <td>40</td>
      <td>#Anticipation #AdventureAwaits</td>
      <td>Twitter</td>
    </tr>
    <tr>
      <th>335</th>
      <td>Thrilled to witness the grandeur of a cultural...</td>
      <td>80</td>
      <td>40</td>
      <td>#Thrill #CulturalCelebration</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>432</th>
      <td>Heartache deepens, a solitary journey through ...</td>
      <td>80</td>
      <td>40</td>
      <td>#Despair #AbyssOfHeartache</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Awe-inspired by the vastness of the cosmos on ...</td>
      <td>80</td>
      <td>40</td>
      <td>#Wonder #StargazingAdventure</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>470</th>
      <td>Dancing on sunshine, each step a celebration o...</td>
      <td>80</td>
      <td>40</td>
      <td>#Joy #SimpleMoments</td>
      <td>Instagram</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9124ccfa-0fa0-459d-a6c2-d850fba26429')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9124ccfa-0fa0-459d-a6c2-d850fba26429 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9124ccfa-0fa0-459d-a6c2-d850fba26429');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-01b2081c-adef-4938-9fae-a26bd0c8dd4c">
      <button class="colab-df-quickchart" onclick="quickchart('df-01b2081c-adef-4938-9fae-a26bd0c8dd4c')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-01b2081c-adef-4938-9fae-a26bd0c8dd4c button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    Top 10 Posts by Retweets:




  <div id="df-ba9cc34f-550f-4054-aa41-38596a140840" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Likes</th>
      <th>Retweets</th>
      <th>Hashtags</th>
      <th>Platform</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>560</th>
      <td>In the serene beauty of a sunset, nature unfol...</td>
      <td>80</td>
      <td>40</td>
      <td>#Tranquility #SunsetBeauty</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>540</th>
      <td>Celebrating a historic victory in the World Cu...</td>
      <td>80</td>
      <td>40</td>
      <td>#Joy #WorldCupTriumph</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>570</th>
      <td>Underneath the city lights, the dancer express...</td>
      <td>80</td>
      <td>40</td>
      <td>#Mesmerizing #NightDancePerformance</td>
      <td>Twitter</td>
    </tr>
    <tr>
      <th>550</th>
      <td>After a series of defeats, the soccer team fac...</td>
      <td>80</td>
      <td>40</td>
      <td>#Disappointment #SoccerDefeats</td>
      <td>Twitter</td>
    </tr>
    <tr>
      <th>510</th>
      <td>At the front row of Adele's concert, each note...</td>
      <td>80</td>
      <td>40</td>
      <td>#Emotion #AdeleConcert</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>530</th>
      <td>Captivated by the spellbinding plot twists, th...</td>
      <td>80</td>
      <td>40</td>
      <td>#Excitement #MoviePremiereThrills</td>
      <td>Twitter</td>
    </tr>
    <tr>
      <th>520</th>
      <td>At a Justin Bieber concert, the infectious bea...</td>
      <td>80</td>
      <td>40</td>
      <td>#Enthusiasm #JustinBieber</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>481</th>
      <td>Surrounded by the colors of joy, a canvas pain...</td>
      <td>80</td>
      <td>40</td>
      <td>#Joy #EndlessSmiles</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Awe-inspired by the vastness of the cosmos on ...</td>
      <td>80</td>
      <td>40</td>
      <td>#Wonder #StargazingAdventure</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>382</th>
      <td>A sense of wonder at the vastness of the cosmo...</td>
      <td>80</td>
      <td>40</td>
      <td>#Wonder #StargazingAdventure</td>
      <td>Instagram</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ba9cc34f-550f-4054-aa41-38596a140840')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ba9cc34f-550f-4054-aa41-38596a140840 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ba9cc34f-550f-4054-aa41-38596a140840');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-525d7415-7f44-4f99-96b7-082bfd72f33d">
      <button class="colab-df-quickchart" onclick="quickchart('df-525d7415-7f44-4f99-96b7-082bfd72f33d')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-525d7415-7f44-4f99-96b7-082bfd72f33d button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




```python
# پلتفرم‌ها بر اساس میانگین Likes و Retweets
engagement_by_platform = df.groupby("Platform")[["Likes", "Retweets"]].mean().sort_values("Likes", ascending=False)

# رسم نمودار
plt.figure(figsize=(10,5))
sns.barplot(data=engagement_by_platform.reset_index(), x="Platform", y="Likes", color="skyblue", label="Likes")
sns.barplot(data=engagement_by_platform.reset_index(), x="Platform", y="Retweets", color="orange", label="Retweets")

plt.title("Average Likes and Retweets per Platform")
plt.ylabel("Average Count")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
```


    
![png](/mnt/data/README_8_0.png)
    


# **Temporal Analysis**
با بررسی ستون Timestamp و روندها را در طول زمان  الگوها، نوسانات یا مضامین تکرارشونده در محتوای شبکه‌های اجتماعی را تحلیل می شود


```python
# اطمینان از تبدیل ستون Timestamp به datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# استخراج ویژگی‌های زمانی
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.day_name()
df["Month"] = df["Timestamp"].dt.month_name()
```


```python
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Hour", palette="coolwarm")
plt.title("Number of Posts by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Posts")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
```

    /tmp/ipython-input-4095864665.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="Hour", palette="coolwarm")



    
![png](/mnt/data/README_11_1.png)
    



```python
plt.figure(figsize=(10,5))
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.countplot(data=df, x="DayOfWeek", order=order, palette="viridis")
plt.title("Number of Posts by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Number of Posts")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
```

    /tmp/ipython-input-3747007276.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="DayOfWeek", order=order, palette="viridis")



    
![png](/mnt/data/README_12_1.png)
    


# **Sentiment Classification**
 در نهایت، با استفاده از یک الگوریتم مناسب، تحلیل احساسات را بر روی ویژگی Text (محتوای تولید شده توسط کاربر) انجام دادم و احساسات کاربران را در دسته‌های
excited، content، calm، angry، sad، disappointed، neutral
  طبقه‌بندی کردم.

### مقدمه‌ای بر روش تحلیل احساسات
اینجا همون قسمتی هست که تو جلسه مصاحبه کاری گفتین باید روش جدید استفاده کنم اولش سعی کردم روش همیشگی یعنی از الگوریتم های نظارت نشده ماشین لرنینگ استفاده کنم که هم زمان بر بود هم پیچیده ولی با کمی جستو جو و تحلیل رفتم سراغ روش جدید یا همان سیا هم یاد گرفتم هم اعمال کردم که توضیحات کاملش هست تو ورد




```python
sia = SentimentIntensityAnalyzer()

# تابعی برای تبدیل نمره به دسته‌بندی احساسات
def classify_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.5:
        return "excited"
    elif score >= 0.2:
        return "content"
    elif score >= 0.05:
        return "calm"
    elif score <= -0.5:
        return "angry"
    elif score <= -0.2:
        return "sad"
    elif score <= -0.05:
        return "disappointed"
    else:
        return "neutral"

# اعمال دسته‌بندی به داده‌ها
df["Predicted_Sentiment"] = df["Text"].apply(classify_sentiment)

# نمایش چند ردیف اول
df[["Text", "Predicted_Sentiment", "Sentiment"]].head(10)
```





  <div id="df-00b17d83-0fef-4bca-8d8b-4f4ac082a7eb" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Predicted_Sentiment</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Enjoying a beautiful day at the park!        ...</td>
      <td>excited</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Traffic was terrible this morning.           ...</td>
      <td>sad</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Just finished an amazing workout! 💪          ...</td>
      <td>excited</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Excited about the upcoming weekend getaway!  ...</td>
      <td>content</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Trying out a new recipe for dinner tonight.  ...</td>
      <td>neutral</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Feeling grateful for the little things in lif...</td>
      <td>excited</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Rainy days call for cozy blankets and hot coc...</td>
      <td>disappointed</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The new movie release is a must-watch!       ...</td>
      <td>neutral</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Political discussions heating up on the timel...</td>
      <td>neutral</td>
      <td>Negative</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Missing summer vibes and beach days.         ...</td>
      <td>sad</td>
      <td>Neutral</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-00b17d83-0fef-4bca-8d8b-4f4ac082a7eb')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-00b17d83-0fef-4bca-8d8b-4f4ac082a7eb button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-00b17d83-0fef-4bca-8d8b-4f4ac082a7eb');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-eb077a8c-f5ba-45a6-8b8f-acaa6177e922">
      <button class="colab-df-quickchart" onclick="quickchart('df-eb077a8c-f5ba-45a6-8b8f-acaa6177e922')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-eb077a8c-f5ba-45a6-8b8f-acaa6177e922 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="Predicted_Sentiment", order=df["Predicted_Sentiment"].value_counts().index, palette="Set2")
plt.title("Predicted Sentiment Categories")
plt.xlabel("Sentiment")
plt.ylabel("Number of Posts")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
```

    /tmp/ipython-input-2756716291.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df, x="Predicted_Sentiment", order=df["Predicted_Sentiment"].value_counts().index, palette="Set2")



    
![png](/mnt/data/README_16_1.png)
    

