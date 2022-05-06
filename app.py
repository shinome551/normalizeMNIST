import tkinter
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np
import torch


class Application(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('Normalize Demo')
        self.pack()
        self.setup()
        self.create_widgets()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load('./model/model.ph').to(self.device)
        
    def create_widgets(self):
        self.clear_button = tkinter.Button(self, text='clear', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=0)

        self.test_canvas = tkinter.Canvas(self, bg='white', width=300, height=300)
        self.test_canvas.grid(row=0, column=1)
        self.test_canvas.bind('<B1-Motion>', self.paint)
        self.test_canvas.bind('<ButtonRelease-1>', self.reset)

        self.test_canvas2 = tkinter.Canvas(self, bg='white', width=300, height=300)
        self.test_canvas2.grid(row=0, column=2)
        self.photo = ImageTk.PhotoImage(self.im)
        self.image_on_canvas = self.test_canvas2.create_image(0, 0, anchor='nw', image=self.photo)

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = 'black'
        self.im = Image.new('RGB', (300, 300), 'white')
        self.draw = ImageDraw.Draw(self.im)

    def clear_canvas(self):
        self.test_canvas.delete(tkinter.ALL)
        self.im = Image.new('RGB', (300, 300), 'white')
        self.draw = ImageDraw.Draw(self.im)
        self.photo = ImageTk.PhotoImage(self.im)
        self.image_on_canvas = self.test_canvas2.create_image(0, 0, anchor='nw', image=self.photo)

    def normalize(self):
        img_normalized = ImageOps.mirror(self.im)
        img_np = np.array(self.im)
        inputs = torch.from_numpy(img_np).rotate(2, 0, 1).unsqueeze(0).div(255).to(self.device)
        outputs, _, _, _ = self.model(inputs)
        outputs = outputs.cpu().mul(255).squeeze(0).rotate(1, 2, 0).numpy()
        img_normalized = Image.fromarray((outputs * 255).astype(np.uint8))

        return img_normalized

    def paint(self, event):
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.test_canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=5.0, fill=paint_color, capstyle=tkinter.ROUND, smooth=tkinter.TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color, width=5)

            self.photo = ImageTk.PhotoImage(self.normalize())
            self.test_canvas2.itemconfig(self.image_on_canvas, image=self.photo)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

root = tkinter.Tk()
app = Application(master=root)
app.mainloop()