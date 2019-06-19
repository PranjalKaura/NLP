import paralleldots
paralleldots.set_api_key("gjZum14a4gPo2PWWYzPHnjOaYxgJIMeGYBOZsHRNlRM")
# for single sentence
text="I am trying to imagine you with a personality."
response=paralleldots.emotion(text)

print(response["emotion"].keys())
print()
print(response["emotion"])



# for multiple sentence as array
# text=["I am trying to imagine you with a personality.","This is shit."]
# response=paralleldots.batch_emotion(text)
# print(response)


#API is free only for 30 days, Start date 13th June 9:00 AM