import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    df_dict = {}
    year_counts = {}
    min_year = 2025
    max_year = 2010
    for df_name, data_path in FILES.items():
        df_dict[df_name] = pd.read_csv(data_path, sep=",")
        # Get min and max year
        if df_dict[df_name].Year.min() < min_year:
            min_year = df_dict[df_name].Year.min()
        if df_dict[df_name].Year.max() > max_year:
            max_year = df_dict[df_name].Year.max()
        # Get year counts
        year_counts[df_name] = df_dict[df_name].Year.value_counts(sort=False)

    years = np.arange(min_year, max_year + 1)
    for df_name, counts in year_counts.items():
        for year in years:
            if year not in counts:
                year_counts[df_name][year] = 0

    yearly_df = pd.DataFrame(year_counts)
    width = 1 / (len(FILES) + 1)
    multiplier = 0

    # Plot figure
    font_size = 14
    plt.style.use('ggplot')
    fig, ax = plt.subplots(layout='constrained')

    for df_name in year_counts.keys():
        offset = width * multiplier
        hatch = "//" if "Image Classification" in df_name else ".."
        rects = ax.bar(yearly_df.index + offset, yearly_df[df_name], width, label=df_name, hatch=hatch)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Number of Publications', fontsize=font_size)
    ax.set_xlabel('Year', fontsize=font_size)
    ax.set_title('Number of Publications in Main Venues per Year', fontsize=font_size + 2)
    ax.set_xticks(years + 1.5*width, years, fontsize=font_size-2)
    ax.legend(loc='upper left', ncols=1, fontsize=font_size)
    plt.savefig("publications_per_year.png")
    # plt.grid(axis='y')
    plt.show()


FILES = {
    "OOD Image Classification Classic": "ood-img-cls-classic.csv",
    "OOD Image Classification VLM": "ood-img-cls-vlm.csv",
    "OOD Object Detection": "ood-obj-det.csv",
    "Open Set Object Detection": "osod-obj-det.csv"
}

if __name__ == '__main__':
    main()