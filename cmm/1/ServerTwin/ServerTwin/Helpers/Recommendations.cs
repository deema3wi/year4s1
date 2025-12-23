namespace ServerTwin.Helpers;

public static class Recommendations
{
	public static string NotEnoughHistory => "Недостатньо історичних даних. Тимчасово вважаємо поточну конфігурацію прийнятною.";
	public static string CPUOverload => "CPU завантажений. Для БД-сервера варто розглянути більше vCPU або додатковий репліка-сервер.";
	public static string CPUNearLimit => "CPU часто наближається до межі. Розглянь збільшення vCPU або додавання ще одного інстанса.";
	public static string RAMOverload => "Використання RAM високе. Для БД краще збільшити обсяг пам'яті або рознести навантаження.";
	public static string RAMNearLimit => "RAM часто завантажена. Можливо, варто збільшити обсяг пам'яті або розподілити сервіси.";
	public static string ServerLowUtilization => "Сервер переважно недовантажений. Можливо, є сенс даунсайзнути інстанс або об'єднати ролі.";
	public static string Ok => "Поточна конфігурація виглядає достатньою. Стеж за піковими навантаженнями, але термінового масштабування не потрібно.";
}