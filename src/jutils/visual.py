"""
Module with graph utilities.
"""
import plotly.express as px
import pandas as pd
from pandas import DataFrame
import numpy as np
import plotly.graph_objects as go


# noinspection GrazieInspection
class Plot:
    """
    Graficar contiene funciones para realizar gráficos predeterminados, tiene un sistema de colores semánticos.
    """

    def __init__(self, colors=None):
        if colors is None:
            self.colors = {
                'femenine': '#ff69b4',
                'masculine': '#7858da',
                'correct': '#1CDB59',
                'incorrect': '#fa1e1e',
                'c1': '#79216d',
                'c2': '#ca1e3f',
                'c3': '#b26116',
                'c4': '#a0933a',
                'c5': '#a3f789'
            }
        else:
            self.colors = colors

        self.color_discrete_sequence = [self.colors[c] for c in ['c1', 'c2', 'c3', 'c4', 'c5']]
        self.color_continuous_scale = ["#fff", self.colors['c1']]

    def with_colors(self, colors):
        new_colors: dict = self.colors
        new_colors.update(colors)
        return Plot(new_colors)

    @staticmethod
    def combine_plots(plot1: go.Figure, plot2: go.Figure) -> go.Figure:
        """
        Une dos graficos en uno.
        Args:
            plot1 (): Primer gráfico
            plot2 (): Segundo gráfico

        Returns:
            Un objeto de plotly con los dos graficos combinados.

        """
        return go.Figure(data=plot1.data + plot2.data)

    def heatmap(self, df: DataFrame, x: str, y: str, aggfunc=len, fill_value=0, **kwargs) -> go.Figure:
        """
        Display a heatmap summarizing the columns x and y by aggfunc.

        Args:
            df (): Dataframe with the data.
            x (): Column to display in the x-axis. (Categorical).
            y (): Column to display in the y-axis. (Categorical).
            aggfunc (): Function to summarize the data.
            fill_value (): Value to replace nulls after the aggregation.
            **kwargs (): Aditional parameters to pass to plotly.express.imshow.

        Returns:
            A Figure that can be displayed with the method show().
        """
        df_heatmap = df[[x, y]].pivot_table(index=y, columns=x, aggfunc=aggfunc, fill_value=fill_value)
        return px.imshow(df_heatmap, color_continuous_scale=self.color_continuous_scale, title=f'{x} vs {y}', **kwargs)

    def box(self, df: DataFrame, y: str, x: str = None, title: str = None, nbins=5, notched=False,
            **kwargs) -> go.Figure:
        """
        Display a boxplot, in reality the x axis is always categorical, but when numerical data is passed
        the x axis is grouped in bins and then displayed.

        Args:
            df (): Dataframe with the data.
            y (): Column to display in the y-axis. (Numerical)
            x (): Column to display in the x-axis. (Categorical | Numerical)
            title (): Plot title.
            nbins: Cant of bins to plot when x is numerical.
            notched (): Set a notch in the plot.
            **kwargs (): Aditional parameters to pass to plotly.express.imshow.

        Returns:
            A Figure that can be displayed with the method show().

        """
        transformed_df = df.copy()
        if x is not None and pd.api.types.is_numeric_dtype(transformed_df[x]):
            minimo = transformed_df[x].min()
            maximo = transformed_df[x].max()
            limites = [round(x) for x in np.linspace(minimo, maximo, nbins)]
            labels = [f'[{lim_min}-{lim_max})'
                      for lim_min, lim_max in zip(limites[0:nbins - 1], limites[1:nbins])]
            bins = pd.cut(transformed_df[x], bins=limites, labels=labels, include_lowest=True)
            transformed_df[x] = bins
        if title is None:
            title = f'Boxplot {x} vs {y}' if x is not None else f'Boxplot {y}'
        return px.box(transformed_df, x=x, y=y, title=title, notched=notched,
                      color_discrete_sequence=self.color_discrete_sequence, **kwargs)

    def histogram(self, df: DataFrame, x: str, nbins: int = None, text_auto=True, **kwargs) -> go.Figure:
        """
        Histogram

        Args:
            df (): Dataframe with the data.
            x (): Column to display in the x-axis. (Categorical | Numerical)
            nbins (): Cant of bins to group the data (If x is categorical, ignores this parameter).
            text_auto (): If False, doesn't show value labels.
            **kwargs (): Aditional parameters to pass to plotly.express.imshow.

        Returns:
            A Figure that can be displayed with the method show().
        """
        return px.histogram(df, x=x, text_auto=text_auto,
                            color_discrete_sequence=self.color_discrete_sequence,
                            nbins=nbins,
                            **kwargs)

    def scatter(self, df: DataFrame, x: str, y: str, color: str = None, correct_incorrect_map=None,
                **kwargs) -> go.Figure:
        """
        Scatter plot

        Args:
            df (): Dataframe with the data.
            x (): Column to display in the x-axis. (Numerical).
            y (): Column to display in the y-axis. (Numerical).
            color (): Column to separate points by color (Categorical).
            correct_incorrect_map (): A dictionary that maps the columns value to the assigned colors.
            **kwargs (): Aditional parameters to pass to plotly.express.imshow.

        Returns:
            A Figure that can be displayed with the method show().

        """
        if correct_incorrect_map is None:
            color_discrete_sequence = self.color_discrete_sequence
            color_discrete_map = None
        else:
            color_discrete_sequence = None
            color_discrete_map = {
                correct_incorrect_map['correct']: self.colors['correct'],
                correct_incorrect_map['incorrect']: self.colors['incorrect']
            }
        return px.scatter(
            data_frame=df,
            x=x,
            y=y,
            color=color,
            color_discrete_sequence=color_discrete_sequence,
            color_discrete_map=color_discrete_map, **kwargs)

    def pyramid(self,
                df: DataFrame,
                x: str,
                nbins: int,
                cat_col: str,
                cat1,
                cat2,
                title: str = None,
                cat1name: str = None,
                cat2name: str = None,
                cat1color=None,
                cat2color=None
                ) -> go.Figure:
        """
        Pyramid plot

        Args:
            df (): Dataframe with the data.
            x (): Column to display in the x-axis. (Numerical).
            nbins (): Cant of bins to group the data.
            cat_col (): Column containing the categories. (Categorical).
            cat1 (): Name of the first category.
            cat2 (): Name of the second category.
            title (): Plot title.
            cat1name (): Display name of the first category in the plot.
            cat2name (): Display name of the second category in the plot.
            cat1color (): Display color of the first category in the plot.
            cat2color (): Display color of the second category in the plot.

        Returns:
            A Figure that can be displayed with the method show().
        """
        cat1name = cat1 if cat1name is None else cat1name
        cat2name = cat2 if cat2name is None else cat2name
        cat1color = self.colors['femenine'] if cat1color is None else cat1color
        cat2color = self.colors['masculine'] if cat2color is None else cat2color
        title = f'Piramide de {cat_col} vs {x}.' if title is None else title

        minimo = df[x].min()
        maximo = df[x].max()
        limites = [round(x) for x in np.linspace(minimo, maximo, nbins)]
        labels = [f'[{lim_min}-{lim_max})'
                  for (lim_min, lim_max) in zip(limites[0:nbins - 1], limites[1:nbins])]

        bins_1 = pd.cut(df.loc[df[cat_col] == cat1][x], bins=limites, labels=labels, include_lowest=True)
        bins_2 = pd.cut(df.loc[df[cat_col] == cat2][x], bins=limites, labels=labels, include_lowest=True)

        bins_1.name = cat1name
        bins_2.name = cat2name

        grouped_1 = bins_1.groupby(bins_1).count()
        grouped_2 = bins_2.groupby(bins_2).count()

        data = pd.DataFrame([grouped_1, grouped_2]).transpose()

        y = data.index
        x_1 = data[cat1name]
        x_2 = data[cat2name] * (-1)
        meta = data[cat2name]

        fig = go.Figure()
        bar1 = go.Bar(y=y, x=x_1, name=cat1name, orientation='h',
                      texttemplate='%{x}', marker_color=cat1color)
        bar2 = go.Bar(y=y, x=x_2, name=cat2name, orientation='h',
                      texttemplate='%{meta}', marker_color=cat2color, meta=meta)
        fig.add_trace(bar1)
        fig.add_trace(bar2)
        fig.update_layout(title=title, barmode='relative', bargap=0.0,
                          bargroupgap=0)
        return fig
