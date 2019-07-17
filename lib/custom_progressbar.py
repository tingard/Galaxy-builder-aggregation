import string
import numpy as np
from IPython.display import display, update_display


class Progress():
    def __init__(self, initial_value=0, min_value=0, max_value=10):
        self.min = min_value
        self.max = max_value
        self.value = initial_value
        self.text = '{} / {}'.format(self.value, self.max)
        styles = ';'.join((
            'display:flex',
            'flex-direction:row',
            'align-items:center',
            'justify-content:center',
            'width:100%'
        ))
        self.container_html = '<div style="{}">{{}}</div>'.format(styles)
        self.text_html = '<h3 style="margin-right:20px">{}</h3>'
        self.bar_html = (
            '<progress value={value} min={min} max={max}'
            ' style="flex-grow:1" />'
        )
        self.display_id = None
        self.show()

    def update(self, new_value, text=None):
        self.value = new_value
        self.text = text if text is not None else '{} / {}'.format(
            self.value, self.max
        )
        if self.display_id is not None:
            update_display(self, display_id=self.display_id)
        return self

    def _repr_html_(self):
        return self.container_html.format(
            self.text_html.format(self.text)
            + self.bar_html.format(**self.__dict__)
        )

    def show(self):
        self.display_id = ''.join(
            np.random.choice(list(string.ascii_lowercase), 30)
        )
        display(self, display_id=self.display_id)
