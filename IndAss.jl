### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 36ab7fc7-960e-4dad-83fd-e3d41b24c8cc
using Pkg;Pkg.add("NORMAL");Pkg.add("PNEUMONIA");Pkg.add("Data");Pkg.add("Plots");Pkg.add("Flux")


# ╔═╡ 26d90424-a038-4dff-abc9-cf823d41bdf9
using NORMAL; using PNEUMONIA; using Data; using Plots

# ╔═╡ ac1e644d-1e8b-4eab-9f70-11f14667a7df
using Flux

# ╔═╡ b4eed947-0df1-40b4-af6b-414a2be443cb
img = Flux.Data.NORMAL.images();

# ╔═╡ d5ec6c38-992c-4cd6-aeec-91702abb4183
lab = Flux.Data.NORMAL.labels();

# ╔═╡ 4503189c-a07f-4a86-9e05-ef8f8a2b976a
typeof(img)

# ╔═╡ 90f2a840-b6f6-463c-abbd-dcb11da82d2a
length(img)

# ╔═╡ 0212afbb-dcfd-4bd1-9ea6-7f421b0cbc02
img[1]
img[1][1,1]
float(img[1][1,1])

# ╔═╡ 552e60a5-16b1-485c-8583-2fd03ad05772
using Plots
plot(plot(img[5]),(plot(img[1019]), (plot(img[1553]), (plot(img[325]), (plot(img[1800]))


# ╔═╡ 482e132a-b7f8-41c1-9851-2cb4c5b65f84
(x_normal, y_normal),(x_pneumonia, y_pneumonia)=Data.load_data()

# ╔═╡ 53414480-d066-11eb-2256-0b40d5e0d32d
x_normal = x_normal.typeof("float32")
x_pneumonia = x_pneumonia.typeof("float32")

# ╔═╡ fbb7f29f-288e-487c-bba8-11e3a9de9b71
x_normal/=255
x_pneumonia /=255

# ╔═╡ 91a476e4-4b1d-4d3e-84d3-c1898e125e92
y_normal = np_utils.to_categorical(y_normal, 10)
y_pneumonia = np_utils.to_categorical(y_pneumonia, 10)


# ╔═╡ 9e479c71-e3a0-43c7-bb6a-6132eae5b1ed
x_normal = x_normal.reshape(x_normal.shape[0], 232,232,1)
x_pneumonia = x_pneumonia.reshape(x_pneumonia.shape[0], 232,232,1)

# ╔═╡ 219eeea7-5908-4b36-a692-c5a8e45ffeb7
model.add(layers.Conv2D(6, kernel_size=(5,5), strides=(1,1), activation='tanh',input_shape=(323,323,1),padding="same"))

# ╔═╡ 537f5084-8c2f-467f-9a09-7353ba76b818
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=(1,1),padding='valid'))

# ╔═╡ 03fd0445-5852-45aa-a831-0c2172b9a30b
model.add(layers.Conv2D(16, kernel_size=(5,5),strides=(1,1),activation='tanh',padding='valid'))


# ╔═╡ 9c6ddf0b-0031-4217-97fe-11294e57b43a
model.add(layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))

# ╔═╡ 9a8dd39e-7d20-47b7-8f8a-92a6beb2d78a
model.add(layers.Conv2D(120, kernel_size=(5,5), strides=(1,1), activation='tanh',padding='valid'))
#flatten the CNN output so that we can connect it with fully connected layers
model.add(layers.Flaten())

# ╔═╡ 548228c2-15cb-4fd1-be66-425bf0be8d0d
model.add(layers.Dense(84, activation='tanh'))

# ╔═╡ c28195a4-f004-4ba3-aaea-9da06f586430
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SDG',metrics=["accuracy"])


# ╔═╡ c81fb88f-fd3a-4cd2-863c-a527a2241f15
pneumonia = model.evaluate(x_pneumonia,y_pneumonia)

# ╔═╡ b6e9ff69-d0e7-4ec6-bc84-efebb5ca7f47
f, ax = plt.subplots()
ax.legend(['Train acc','Validation acc'], loc=0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('acc')
ax.set_ylabel('Loss')

# ╔═╡ cde2592d-0953-4318-88f3-3f64aae6f70b
import model

# ╔═╡ 2916b66f-6575-4134-97da-1412d4d19c41
model = Sequential()

# ╔═╡ Cell order:
# ╠═36ab7fc7-960e-4dad-83fd-e3d41b24c8cc
# ╠═26d90424-a038-4dff-abc9-cf823d41bdf9
# ╠═ac1e644d-1e8b-4eab-9f70-11f14667a7df
# ╠═b4eed947-0df1-40b4-af6b-414a2be443cb
# ╠═d5ec6c38-992c-4cd6-aeec-91702abb4183
# ╠═4503189c-a07f-4a86-9e05-ef8f8a2b976a
# ╠═90f2a840-b6f6-463c-abbd-dcb11da82d2a
# ╠═0212afbb-dcfd-4bd1-9ea6-7f421b0cbc02
# ╠═552e60a5-16b1-485c-8583-2fd03ad05772
# ╠═482e132a-b7f8-41c1-9851-2cb4c5b65f84
# ╠═53414480-d066-11eb-2256-0b40d5e0d32d
# ╠═fbb7f29f-288e-487c-bba8-11e3a9de9b71
# ╠═91a476e4-4b1d-4d3e-84d3-c1898e125e92
# ╠═9e479c71-e3a0-43c7-bb6a-6132eae5b1ed
# ╠═cde2592d-0953-4318-88f3-3f64aae6f70b
# ╠═2916b66f-6575-4134-97da-1412d4d19c41
# ╠═219eeea7-5908-4b36-a692-c5a8e45ffeb7
# ╠═537f5084-8c2f-467f-9a09-7353ba76b818
# ╠═03fd0445-5852-45aa-a831-0c2172b9a30b
# ╠═9c6ddf0b-0031-4217-97fe-11294e57b43a
# ╠═9a8dd39e-7d20-47b7-8f8a-92a6beb2d78a
# ╠═548228c2-15cb-4fd1-be66-425bf0be8d0d
# ╠═c28195a4-f004-4ba3-aaea-9da06f586430
# ╠═c81fb88f-fd3a-4cd2-863c-a527a2241f15
# ╠═b6e9ff69-d0e7-4ec6-bc84-efebb5ca7f47