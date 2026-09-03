"""Placeholder panels for sections not yet implemented."""
import panel as pn
import param


class StubPanel(param.Parameterized):
    section_name = param.String(default="Section")

    def __init__(self, app, section_name: str, **params):
        super().__init__(section_name=section_name, **params)
        self.app = app

    def view(self):
        return pn.Column(
            pn.pane.HTML(
                f'<div style="color:#556070;font-size:14px;padding:24px;">'
                f'<b style="color:#8a9bb5">{self.section_name}</b><br><br>'
                f'This panel is not yet implemented.'
                f'</div>'
            )
        )
