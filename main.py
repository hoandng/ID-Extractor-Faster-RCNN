from src.inference import CCCDPipeline

pipe   = CCCDPipeline(
    card_model   = "weights/card/best.pth",
    corner_model = "weights/corner/best.pth",
    field_model  = "weights/field/best.pth",
)

result = pipe.run("dataset/test/cccd_mt.jpg")

if result["status"] == "success":
    for field, text in result["data"].items():
        print(f"{field}: {text}")

else:
    print(result["status"])